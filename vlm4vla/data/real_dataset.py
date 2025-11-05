import json
import os
from random import shuffle

import numpy as np
from functools import partial

import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from PIL import Image
from dataclasses import dataclass
from vlm4vla.data.base_task_dataset import BaseTaskDataset
from typing import Callable, Any, Dict, Sequence
from torchvision.transforms.functional import to_pil_image


class RealDataset(BaseTaskDataset):
    # need to read whole dataset into memory before loading to gpu
    # predict future 10 steps image
    def __init__(
        self,
        model_name,
        data_dir,
        window_size,
        fwd_pred_next_n,
        norm_action,
        **kwargs,
    ):
        self.dataset_path = data_dir
        self.window_size = window_size
        self.model_name = model_name
        self.fwd_pred_next_n = fwd_pred_next_n
        self.norm_action = norm_action
        self.episodes, self.instructions, (self.actions, _) = load_json_data(data_dir)
        self.length_episodes = np.cumsum([len(i) for i in self.episodes])
        self.length_episodes = {i: self.length_episodes[i] for i in range(len(self.length_episodes))}
        self.future_step = fwd_pred_next_n
        assert self.window_size == 1
        kwargs["task_type"] = "action"
        super().__init__(model_name=model_name, **kwargs)
        print("Formatting Real data")

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, index):
        # while try get_raw_items, if error, try get_raw_items(index+1) until success
        while True:
            try:
                data_dict = self.get_raw_items(index)
                break
            except Exception as e:
                # print(f"Error in get_raw_items: {e}")
                # print(f"Trying get_raw_items(index+1)")
                index = (index + 1) % len(self.actions)

        future_index = self.get_future_index(index, future_step=self.future_step)
        # data_dict_future = self.get_raw_items(future_index)
        # assert data_dict['input_ids'] == data_dict_future['input_ids']
        # data_dict['images_static_future'] = data_dict_future['images_static']
        # data_dict['images_gripper_future'] = data_dict_future['images_gripper']
        # data_dict['observation']['image_primary_future'] = data_dict_future['observation']['image_primary']
        if index == future_index:

            actions = torch.tensor(self.actions[index:future_index + 1])
        else:
            actions = torch.tensor(self.actions[index:future_index])  # n,7 （n,2,7)
        offset = self.future_step - actions.shape[0]
        if offset > 0:
            chunk_mask = torch.cat([torch.ones(size=(actions.shape[0],)), torch.zeros(size=(offset,))], dim=0)
            pad_tube = torch.zeros(size=(offset, actions.shape[-2], actions.shape[-1]), dtype=actions.dtype)
            pad_tube[:, :, -1] = actions[-1, :, -1]  # gripper state of last action is repeated
            actions = torch.cat([actions, pad_tube], dim=0)
        else:
            chunk_mask = torch.ones(size=(actions.shape[0],))
        data_dict['action_chunk'] = actions.unsqueeze(
            0)  # (self.future_step, 7) (1,10,7) add self.window_size dimension
        data_dict['chunk_mask'] = chunk_mask.unsqueeze(0)
        return data_dict

    def get_raw_items(self, index):
        episode_idx, idx = self.get_episode_idx(index)
        episode = self.episodes[episode_idx]
        # sequence_length * epi[0],epi[1],...
        # image_gripper = Image.open(episode[idx]['rgb_head']).convert('RGB')  # hwc,255
        # self.image_fn 需要输入PIL.Image对象
        image_tensors = self.image_fn([Image.open(episode[idx]['rgb_head']).convert('RGB')], static=True)
        # save image_tensors to png
        # 确保张量值域在0-255范围内，避免颜色反转
        # tensor_to_save = image_tensors.squeeze()
        # if tensor_to_save.max() <= 1.0:  # 如果张量值域在[0,1]
        #     tensor_to_save = tensor_to_save * 255.0
        # tensor_to_save = torch.clamp(tensor_to_save, 0, 255).to(torch.uint8)
        # to_pil_image(tensor_to_save).save(f"image_tensors.png")
        # proprio = torch.tensor(self.states[index])

        instruction = self.instructions[episode_idx]
        data_dict = dict(
            rgb=image_tensors,  # add self.window_size dimension
            # rel_state=proprio.unsqueeze(0),  # add self.window_size dimension
            instruction=instruction,
        )
        return data_dict

    def get_episode_idx(self, index):
        for i, x in self.length_episodes.items():
            if index < x:
                episode_idx = i
                idx = index - self.length_episodes[episode_idx - 1] if i != 0 else index
                return episode_idx, idx
        raise ValueError(f"Index {index} out of range")

    def get_future_index(self, index, future_step=10):
        for i, x in self.length_episodes.items():
            if index < x:
                if index + future_step < x:
                    return index + future_step  # future index is in the same episode
                else:
                    return self.length_episodes[i] - 1  # future index is in the next episode, use the last frame
        raise ValueError(f"Index {index} out of range")

    def init_batch_transform(self):
        # no need to implement batch transform function
        return None

    def init_collater_fn(self):

        return RealDualArmPaddedCollator(
            window_size=self.window_size,
            fwd_pred_next_n=self.fwd_pred_next_n,
            text_fn=self.text_fn,
            model=self.model if hasattr(self, "model") else None)


@dataclass
class RealDualArmPaddedCollator:
    fwd_pred_next_n: int
    window_size: int
    text_fn: Callable
    model: Any

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        (
            image_tensors,
            # rel_state,
            instruction,
            action_chunk,
            chunk_mask,
        ) = tuple([instance[key] for instance in instances] for key in (
            "rgb",
            # "rel_state",
            "instruction",
            "action_chunk",
            "chunk_mask",
        ))
        input_ids, attention_mask = self.text_fn(instruction)
        seq_len = self.window_size
        if isinstance(input_ids, list) and isinstance(input_ids[0], str):
            # 是句子，没被tokenize
            image_inputs = []
            text = []
            for i in range(len(image_tensors)):
                for j in range(seq_len):
                    image_inputs.append(to_pil_image(image_tensors[i][j]))
                    text.append(input_ids[i])
            image_inputs = self.model.process_vision_info(image_inputs)
            self.model.tokenizer.padding_side = "right"
            inputs = self.model.processor(
                text=text,
                images=image_inputs,
                return_tensors="pt",
                padding=True,
                videos=None,
            )
            input_ids = inputs
            attention_mask = inputs["attention_mask"]

        image_tensors = torch.stack(image_tensors)
        # action_tensors = torch.stack(action_chunk).squeeze()
        # action_mask = torch.stack(chunk_mask).squeeze()

        # if not self.organize_type == "segment":
        #     action_chunk = action_tensors[:, -self.fwd_pred_next_n:]
        #     action_chunk_mask = action_mask[:, -self.fwd_pred_next_n:]
        # else:
        #     action_chunk = torch.stack(action_chunk)
        #     action_chunk_mask = torch.stack(action_chunk_mask)
        # import pdb; pdb.set_trace()
        action_chunk = torch.stack(action_chunk)
        action_chunk_mask = torch.stack(chunk_mask)

        output = {
            "rgb": image_tensors,
            "hand_rgb": None,
            "fwd_rgb_chunck": None,
            "fwd_hand_rgb_chunck": None,
            "fwd_mask": None,
            "text": input_ids,
            "text_mask": attention_mask,
            # "action": action_tensors,
            # "action_mask": action_mask,
            "action_chunck": action_chunk,
            "chunck_mask": action_chunk_mask,
            "instr_and_action_ids": None,
            "instr_and_action_labels": None,
            "instr_and_action_mask": attention_mask,
        }
        return output


def load_json_data(dataset_path):
    episodes = []
    instructions = []
    actions = []
    states = []
    with open(os.path.join(dataset_path, 'states_actions.json'), 'r') as f:
        dataset = json.load(f)
    error_count = 0
    error_count_no_head = 0
    for epi in dataset:
        frames = []
        for frame in epi["steps"]:
            # path_head = "/mnt/zjk/jianke_z/"
            # path_head = "/cephfs/shared/zjk_processed_data"
            # path_tale1 = "/".join(frame["wrist_1"].split("/")[3:])  # /cephfs/shared
            # path_tale2 = "/".join(frame["wrist_2"].split("/")[3:])  # /cephfs/shared
            # frames.append({'rgb_gripper': path_head+path_tale1, 'rgb_static': path_head+path_tale2})
            # print(frame["images"])
            # print(frame["images"])
            # if len(episodes) > 0:
            #     print(episodes[-1][-1])
            if len(frame["images"]) < 3:
                error_count += 1
                # continue

            # if len(frame["images"]) == 0:
            #     # 没存图的直接用上一个
            #     frame_path = os.path.join(dataset_path, "/".join(frames[-1]["rgb_head"].split("/")[:-1]))
            # else:
            if len(frame["images"]) == 0:
                frame_path = None
            else:
                frame_path = os.path.join(dataset_path, "/".join(frame["images"][0].split("/")[:-1]))
                if not os.path.exists(os.path.join(frame_path, "head.jpeg")):
                    error_count_no_head += 1
                    frame_path = None
                    # print(frame_path)
                    # frame_path = os.path.join(dataset_path, "/".join(frames[-1]["rgb_head"].split("/")[:-1]))
                    # # 寻找向后寻找最近的有rgb_head的帧，frames中存的是之前的
            if frame_path is None:
                frames.append(None)
            else:
                frames.append({
                    'rgb_head': os.path.join(frame_path, "head.jpeg"),
                    # 'rgb_head': "/mnt/zjk/jianke_z/x2w_data_0924/episode_000/000000/head.jpeg",
                    'rgb_gripper_left': os.path.join(frame_path, "left.jpeg"),
                    'rgb_gripper_right': os.path.join(frame_path, "right.jpeg"),
                })
            actions.append(frame["action"])
            states.append(frame["joint_state"])

        instructions.append(epi['steps'][0]["language_instruction"].strip())
        episodes.append(frames)
    print(f"error_count: {error_count}")
    print(f"error_count_no_head: {error_count_no_head}")
    return episodes, instructions, (actions, states)
