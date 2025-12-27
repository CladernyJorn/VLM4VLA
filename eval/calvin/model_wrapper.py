import json
import os.path
from copy import deepcopy
import torch
from PIL import Image
from typing import Literal
import numpy as np
import functools

from lightning.pytorch.trainer import Trainer

from eval.calvin.eval_utils import init_trainer_config, euler2rotm, rotm2euler
from vlm4vla.train.base_trainer import BaseTrainer
from vlm4vla.utils.model_utils import build_tokenizer
from vlm4vla.data.datamodule.gr_datamodule import GRDataModule
from vlm4vla.data.data_utils import get_text_function
from vlm4vla.data.data_utils import (
    preprocess_image,
    tcp_to_world_frame,
)
from queue import Queue

fwd_decay_ratio = 1


class CustomModel:
    # model option
    def __init__(
        self,
        ckpt_path,
        configs,
        device,
        save_dir=None,
        raw_calvin=True,
        debug=False,
        action_ensemble=False,
    ):
        self.model = BaseTrainer(configs=configs)
        self.init_config(ckpt_path, configs, device, save_dir, raw_calvin, debug)
        # self.model.model.lm_head.window_size = 1

    def init_config(self, ckpt_path, configs, device, save_dir=None, raw_calvin=False, debug=False):
        ### load and convert checkpoint
        self.debug = debug
        if configs["model"] == "kosmos":
            import transformers

            package_dir = transformers.__path__[0]
            os.system("cp tools/modeling_kosmos2.py {}/models/kosmos2/modeling_kosmos2.py".format(package_dir))

        if not self.debug:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            if "state_dict" in ckpt:
                new_state_dict = ckpt["state_dict"]
            elif "model_state_dict" in ckpt:
                new_state_dict = ckpt["model_state_dict"]
            else:
                raise KeyError("no checkpoint dict in the loaded pretrain parameters")

            new_state_dict = self.convert_old_state_dict(new_state_dict)
            # print(self.model.model.model.dtype) # bf16
            print(f"CKPT Loading from {ckpt_path}")
            msg = self.model.load_state_dict(new_state_dict, strict=False)
            # print(self.model.model.model.dtype) # bf16
            print(f"CKPT Loaded from {ckpt_path}\n {msg}")

            ckpt_dir = os.path.dirname(ckpt_path)
            ckpt_name = os.path.basename(ckpt_path)
            save_dir = ckpt_dir if save_dir is None else save_dir
            # print(f"save_dir: {save_dir}")
            # print(f"ckpt_dir: {ckpt_dir}")
            # print(f"ckpt_name: {ckpt_name}")
            load_info_path = os.path.join(save_dir, f"{ckpt_name}_loading_msg.json")
            if os.path.exists(load_info_path):
                os.system(f"rm {load_info_path}")
            with open(load_info_path, "w") as f:
                _info = {
                    "missing_keys": msg.missing_keys,
                    "unexpected_keys": msg.unexpected_keys,
                }
                json.dump(_info, f, indent=2)
                print(f"Model loading msg is updated to: {load_info_path}")

        self.configs = configs

        dtype = torch.float32
        if self.configs["trainer"]["precision"] == "bf16":
            dtype = torch.bfloat16
        elif self.configs["trainer"]["precision"] == "fp16":
            dtype = torch.float16
        self.dtype = dtype
        self.act_head_configs = self.configs["act_head"]
        self.raw_calvin = raw_calvin
        self.tcp_rel = self.configs.get("tcp_rel", False)

        print(f"raw action: {self.raw_calvin}")

        self.device = device
        self.policy = self.model
        self.policy = self.policy.to(self.dtype)
        # self.policy = self.policy.float()
        if self.device:
            self.policy.to(self.device)
        self.policy.eval()

        if not hasattr(self.policy.model, "lm_head") and hasattr(self.policy.model, "act_head"):
            self.policy.model.lm_head = self.policy.model.act_head

        self.tokenizer = build_tokenizer(self.configs["tokenizer"])

        self.window_size = configs["window_size"]
        self.fwd_pred_next_n = configs["fwd_pred_next_n"]
        self.act_step = self.fwd_pred_next_n + 1
        self.seq_len = self.configs["seq_len"]
        self.use_hand_rgb = self.configs["use_hand_rgb"]

        if hasattr(self, "policy_setup"):
            data_mix = "bridge" if self.policy_setup == "widowx_bridge" else "rt_1"
            configs["train_dataset"]["data_mix"] = data_mix
            configs["val_dataset"]["data_mix"] = data_mix

        image_preprocess = self.model.model.image_processor
        self.image_preprocess = functools.partial(
            preprocess_image,
            image_processor=image_preprocess,
            model_type=configs["model"],
        )
        if configs["model"] == "qwen25vl":
            if "longsequence" in ckpt_path:
                qwen25_seq_id = 0
            elif "shortsequence" in ckpt_path:
                qwen25_seq_id = 1
            else:
                # raise ValueError("Unknown Qwen2.5 checkpoint")
                qwen25_seq_id = 0  # default to long sequence
        else:
            qwen25_seq_id = None

        if 'prompt' in configs:
            robot_prompt = configs["prompt"]
        else:
            robot_prompt = None
        print('robot_prompt', robot_prompt)
        self.text_preprocess = get_text_function(self.model.model.tokenizer, configs["model"], qwen25_seq_id, robot_prompt)
        
        self.action_space = self.configs["act_head"].get("action_space", "continuous")

        print(f"Evaluating checkpoint {ckpt_path}")

        self.rgb_list = []
        self.hand_rgb_list = []
        self.action_hist_list = []
        self.rollout_step_counter = 0

        self.vision_queue = Queue(maxsize=self.window_size)
        self.vision_gripper_queue = Queue(maxsize=self.window_size)
        self.action_queue = Queue(maxsize=self.window_size - 1)

    def ensemble_action(self, action):
        if action.ndim >= 3:
            action = action.squeeze()

        if action.ndim == 1:
            action = action.unsqueeze(0)

        self.action_hist_list.append(action)

        act_cache = []
        # self.fwd_pred_next_n = 1
        max_len = self.fwd_pred_next_n
        max_len = 1
        # if self.tcp_rel:
        #     max_len = 1
        while len(self.action_hist_list) > max_len:
            self.action_hist_list.pop(0)

        idx = 0
        for act in self.action_hist_list[::-1]:
            # print(act.shape)
            act_cache.append(act[idx])
            idx += 1

        act_cache = torch.stack(act_cache, dim=0)

        weights = torch.tensor([fwd_decay_ratio**i for i in range(len(act_cache))])
        weights = weights / weights.sum()

        weighted_act = (act_cache * weights.unsqueeze(1)).sum(dim=0)

        return weighted_act

    @staticmethod
    def convert_old_state_dict(state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_k = k.replace("module.", "")
            else:
                new_k = k

            if not new_k.startswith("model."):
                new_k = "model." + new_k

            new_state_dict[new_k] = state_dict[k]
        return new_state_dict

    def _get_default_calvin_config(self):
        return {
            "type": "DiskCalvinDataset",
            "data_dir": "CALVIN/task_ABCD_D/val",
            "c_act_scaler": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }

    def add_element_to_queue(self, q: Queue, element):
        while q.qsize() >= q.maxsize:
            q.get()
        q.put(element)

    def get_history(self, q: Queue, pad: Literal["zero", "first"] = "zero"):
        queue_list = list(q.queue)
        if len(queue_list) == 0:
            return queue_list, None
        history_type = self.configs["act_head"].get("history_type", "pre")
        if history_type == "pre":
            pad_len = 0
        else:
            raise ValueError(f"Unsupported history type {history_type}")
        element = queue_list[0]
        if pad == "zero":
            if isinstance(element, torch.Tensor):
                element = torch.zeros_like(element)
            elif isinstance(element, np.ndarray):
                element = np.zeros_like(element)
            else:
                raise ValueError("This type is not supported")
            queue_list = [element for _ in range(pad_len)] + queue_list
        else:
            if isinstance(element, torch.Tensor):
                pad_list = [element.clone() for _ in range(pad_len)]
            elif isinstance(element, np.ndarray):
                pad_list = [deepcopy(element) for _ in range(pad_len)]
            queue_list = pad_list + queue_list
        pad_mask = np.ones(q.maxsize, dtype=bool)
        pad_mask[:pad_len] = False
        return queue_list, pad_mask

    def preprocess(self, obs, lang, mode="continuous"):
        # preprocess image
        image = obs["rgb_obs"]["rgb_static"]
        image = Image.fromarray(image)
        image_x = self.image_preprocess([image]).unsqueeze(0)

        gripper_x = None
        if "rgb_gripper" in obs["rgb_obs"]:
            gripper = obs["rgb_obs"]["rgb_gripper"]
            gripper = Image.fromarray(gripper)
            gripper_x = self.image_preprocess([gripper]).unsqueeze(0)
            gripper_x = gripper_x.to(self.device).to(self.dtype)

        if self.configs["act_head"].get("history_type", "post") == "pre":
            self.add_element_to_queue(self.vision_queue, image_x)
            image_x, _ = self.get_history(self.vision_queue, pad="first")
            image_x = torch.concatenate(image_x, dim=1)

            if gripper_x is not None:
                self.add_element_to_queue(self.vision_gripper_queue, gripper_x)
                gripper_x, _ = self.get_history(self.vision_gripper_queue, pad="first")
                gripper_x = (torch.concatenate(gripper_x, dim=1).to(self.device).to(self.dtype))

        text_x, mask = self.text_preprocess([lang])

        # return (
        #     image_x.to(self.device).to(self.dtype),
        #     gripper_x,
        #     text_x.to(self.device),
        #     mask.to(self.device),
        # )
        return (
            image_x.to(self.dtype),
            gripper_x,
            text_x,
            mask,
        )

    def step(self, obs, goal, execute_step):
        # execute_step: number of steps to execute the same action chunk
        # if execute_step==1, equals execute_chunk= False, mostly slow
        # if execute_step>10, equals execute_chunk= True, mostly fast
        assert execute_step >= 1 and execute_step <= self.fwd_pred_next_n
        if self.rollout_step_counter % execute_step == 0:
            # get new action chunk via model inference
            """Step function."""
            input_dict = dict()
            image_x, gripper_x, text_x, mask = self.preprocess(obs, goal, self.action_space)

            input_dict["rgb"] = image_x
            input_dict["hand_rgb"] = gripper_x
            input_dict["text"] = text_x
            input_dict["text_mask"] = mask

            with torch.no_grad():
                action = self.policy.inference_step(input_dict)["action"]

            if self.action_space != "discrete":
                # print(action)
                if action[0].ndim == action[1].ndim + 1:
                    action = (action[0], action[1].unsqueeze(2))
                action = torch.cat(
                    [action[0], (torch.nn.functional.sigmoid(action[1]) > 0.5).float()],
                    dim=-1,
                )

            # action = action[0, 0, 0] # batch, seq_len, chunck_idx

            if isinstance(action, tuple):
                action = torch.cat([action[0], action[1]], dim=-1)

            if isinstance(action, np.ndarray):
                action = torch.from_numpy(action)

            if action.ndim == 2:
                action = action.unsqueeze(1)

            if action.ndim == 3:
                action = action.unsqueeze(1)

            action = action.detach().cpu()

            if self.tcp_rel:
                robot_obs = (
                    torch.from_numpy(obs["robot_obs"]).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(
                        1, 1, self.fwd_pred_next_n, 1))
                action = tcp_to_world_frame(action, robot_obs)
            # print(action.shape)  # (1,1,10,7)
            if execute_step == 1:
                action = self.ensemble_action(action)  # (7,)
            else:
                action = action.squeeze()
                assert len(self.action_hist_list) == 0
                assert action.shape[0] == self.fwd_pred_next_n
                self.action_hist_list = list(action)[:execute_step]
                action = self.action_hist_list.pop(0)  # (7,)
        else:
            action = self.action_hist_list.pop(0)

        if isinstance(action, torch.Tensor):
            action = action.squeeze()
            if action.ndim == 2:
                action = action[0]
            # action = action.numpy()

        if self.configs.get("use_mu_law", False):
            from vlm4vla.data.data_utils import inverse_mu_law_companding

            action = inverse_mu_law_companding(action, self.configs.get("mu_val", 255), maintain_last=True)

        if self.configs.get("norm_action", False):
            from vlm4vla.data.data_utils import unnoramalize_action

            if isinstance(action, tuple):
                action = (
                    unnoramalize_action(action[0], self.configs["norm_min"], self.configs["norm_max"]),
                    action[1],
                )
            else:
                action = unnoramalize_action(action, self.configs["norm_min"], self.configs["norm_max"])

        if self.action_space == "discrete":
            # action[-1] = 1 if action[-1] > 0 else -1
            pass
        else:
            if self.raw_calvin:
                action[-1] = (action[-1] - 0.5) * 2
            else:
                state = obs["robot_obs"]  # (15,)
                xyz_state = state[:3]
                rpy_state = state[3:6]
                rotm_state = euler2rotm(rpy_state)
                rel_action = action.numpy()
                _c_rel_action = rel_action[:6]
                xyz_action = _c_rel_action[:3]
                rpy_action = _c_rel_action[3:6]
                gripper_action = rel_action[6]
                rotm_action = euler2rotm(rpy_action)
                xyz_next_state = xyz_state + rotm_state @ xyz_action
                rotm_next_state = rotm_state @ rotm_action
                rpy_next_state = rotm2euler(rotm_next_state)

                action = action.numpy()
                action[:3] = xyz_next_state - xyz_state
                action[3:6] = rpy_next_state - rpy_state
                action[:6] *= [50.0, 50.0, 50.0, 20.0, 20.0, 20.0]
                action[-1] = (gripper_action - 0.5) * 2
                action = torch.from_numpy(action)

        self.rollout_step_counter += 1
        action[-1] = 1 if action[-1] > 0 else -1
        # print(f"step {self.rollout_step_counter} action {action}")
        return action

    def reset(self):
        if hasattr(self.model.model, "lm_head"):
            self.model.model.lm_head.hidden_state = None
            self.model.model.lm_head.history_memory = []
        if hasattr(self.model.model, "act_head"):
            self.model.model.act_head.hidden_state = None
            self.model.model.act_head.history_memory = []

        self.rgb_list = []
        self.hand_rgb_list = []
        self.rollout_step_counter = 0
        self.action_hist_list = []

        while not self.vision_queue.empty():
            self.vision_queue.get()
        while not self.vision_gripper_queue.empty():
            self.vision_gripper_queue.get()
        while not self.action_queue.empty():
            self.action_queue.get()
