from dataclasses import dataclass
import numpy as np
from PIL import Image
from typing import Any, Dict, Callable, List, Tuple, Union, Literal, Optional, Sequence

import torch
import torch.nn.functional as F

from transformers import PreTrainedTokenizerBase

from vlm4vla.data.base_task_dataset import BaseTaskDataset, IGNORE_INDEX
from torchvision.transforms.functional import to_pil_image
from vlm4vla.data.data_utils import (
    get_tensor_chunk,
    mu_law_companding,
    normalize_action,
    pad_sequences,
    regularize_action,
)


@dataclass
class ActionPredictionBatchTransform:
    """
    Transform one item of dataset
    """

    model_name: str
    tokenizer: PreTrainedTokenizerBase
    text_fn: Callable
    image_fn: Callable[[List[Image.Image]], torch.Tensor]

    window_size: int
    fwd_pred_next_n: int
    predict_stop_token: bool

    organize_type: str
    image_history: bool
    action_history: bool
    discrete: bool
    action_tokenizer: Any
    special_history_id: int
    mode: str

    norm_action: bool
    norm_min: float
    norm_max: float
    x_mean: float
    x_std: float
    regular_action: bool
    use_mu_law: bool
    min_action: float
    max_action: float

    @staticmethod
    def refine_action_at_gripper_dim(action: Union[np.ndarray, torch.Tensor], value: int = -1, status: bool = False):
        """
        make the open gripper action state as value (0 or 1)
        """
        # 确保action最后一维大小为7
        assert action.shape[-1] == 7, "The action dimension must be 7 if refine action at gripper dim"
        if isinstance(action, np.ndarray):
            action = action.copy()
        elif isinstance(action, torch.Tensor):
            action = action.clone()
        else:
            raise TypeError("The type of action must be ndarray or tensor")
        gripper_action = action[..., -1]
        if status:
            gripper_action[gripper_action == 1] = value
        else:
            gripper_action[gripper_action != 1] = value
        return action

    def convert_image(
        self,
        images: Optional[np.ndarray],
        image_mask: torch.Tensor,
        static: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if images is None:
            return None, None, None

        # Image.fromarray(images[0]).save("test_bridge.png")
        if not self.image_history:
            image_tensors = self.image_fn([Image.fromarray(images[self.window_size - 1])], static=static)
            return image_tensors, None, None

        image_tensors = self.image_fn([Image.fromarray(each_image) for each_image in images], static=static)

        # you can't get chunk image in the segment dataset because segment dataset will padding in the left side
        if self.organize_type == "segment":
            return image_tensors, None, None

        left_pad_index = self.window_size - image_mask[:self.window_size].sum()
        image_tensors[:left_pad_index] = image_tensors[left_pad_index]

        # this chunk is to predict next fwd_pred_next_n images, it is based on one image, so we need to skip the first one which including image0
        image_chunk = get_tensor_chunk(image_tensors, self.fwd_pred_next_n)[1:]
        image_chunk_mask = get_tensor_chunk(image_mask, self.fwd_pred_next_n)[1:]

        image_tensors = image_tensors[:self.window_size]
        return image_tensors, image_chunk, image_chunk_mask

    def convert_action(self, action: np.ndarray, action_mask: torch.Tensor):
        # ACTION
        if self.mode == "train":
            # the act step set to fwd_pred_next_n + 1, it will get one more action which we should drop it
            action = action[:-1]
            action_mask = action_mask[:-1]
        else:
            # in inference, this mask will be give by the image mask, which is one more than action action (we have current image but don't know current action)
            action_mask = action_mask[:-1]

        if self.norm_action:
            action = normalize_action(action, self.norm_min, self.norm_max, maintain_last=True)
        if self.regular_action:
            action = regularize_action(action, self.x_mean, self.x_std)
        if self.use_mu_law:
            action = mu_law_companding(action)
        if action.shape[-1] == 7:
            action = self.refine_action_at_gripper_dim(action, value=0)
        action = torch.tensor(action)
        if self.mode != "train":
            return action, action_mask, None, None

        action_chunk = get_tensor_chunk(action, self.fwd_pred_next_n)
        action_chunk_mask = get_tensor_chunk(action_mask, self.fwd_pred_next_n)
        return action, action_mask, action_chunk, action_chunk_mask

    def get_right_pad_len(self, action_chunk_mask: np.ndarray, action_dim: int):
        right_chunk_mask = action_chunk_mask[-self.fwd_pred_next_n:]
        return (right_chunk_mask.shape[0] - right_chunk_mask.sum()) * action_dim

    def wrap_instruction_and_action_interleave_for_continuous_action(self, task_description: str, action: torch.Tensor,
                                                                     action_mask: torch.Tensor):
        if self.mode == "train":
            assert action.shape[0] == self.window_size + self.fwd_pred_next_n - 1

        action_mask = action_mask.bool()
        if action.shape[-1] == 7:
            action = self.refine_action_at_gripper_dim(action, value=self.min_action, status=False)
            action = self.refine_action_at_gripper_dim(action, value=self.max_action, status=True)
        action = action.flatten()
        all_input_ids = task_description
        return all_input_ids, None, None

    def __call__(
        self,
        task_description: str,
        action: np.ndarray,
        episode_mask: np.ndarray,
        images: np.ndarray,
        gripper_images: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Converts an item to the format expected by collator/models."""
        episode_mask = torch.tensor(episode_mask)

        # Pad in Image and action tensors
        image_tensors, image_chunk, image_chunk_mask = self.convert_image(images, episode_mask)
        gripper_image_tensors, gripper_image_chunk, _ = self.convert_image(gripper_images, episode_mask, static=False)

        # ACTION TENSORS
        action, action_mask, action_chunk, action_chunk_mask = self.convert_action(action, episode_mask)

        # INPUT IDS (OPTIONAL WITH DISCRETE ACTION IDS)
        if self.organize_type == "interleave" and not self.discrete:
            (
                input_ids,
                labels,
                attention_mask,
            ) = self.wrap_instruction_and_action_interleave_for_continuous_action(task_description, action, action_mask)
        else:
            raise TypeError("The organize type must be interleave or segment")

        return dict(
            image_tensors=image_tensors,
            image_chunk=image_chunk,
            image_chunk_mask=image_chunk_mask,
            gripper_image_tensors=gripper_image_tensors,
            gripper_image_chunk=gripper_image_chunk,
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            action_tensors=action,
            action_mask=action_mask,
            action_chunk=action_chunk,
            action_chunk_mask=action_chunk_mask,
        )


@dataclass
class ActionPredictionPaddedCollator:
    pad_token_id: int
    fwd_pred_next_n: int
    window_size: int
    organize_type: str
    discrete: bool = False
    text_fn: Callable = None
    model: Any = None

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        (
            image_tensors,
            image_chunk,
            image_chunk_mask,
            gripper_image_tensors,
            gripper_image_chunk,
            input_ids,
            labels,
            attention_mask,
            action_tensors,
            action_mask,
            action_chunk,
            action_chunk_mask,
        ) = tuple([instance[key] for instance in instances] for key in (
            "image_tensors",
            "image_chunk",
            "image_chunk_mask",
            "gripper_image_tensors",
            "gripper_image_chunk",
            "input_ids",
            "labels",
            "attention_mask",
            "action_tensors",
            "action_mask",
            "action_chunk",
            "action_chunk_mask",
        ))
        if self.organize_type == "interleave" and not self.discrete:
            assert attention_mask[0] is None
            assert labels[0] is None
            assert isinstance(input_ids[0], str)
            input_ids, attention_mask = self.text_fn(input_ids)
            labels = None
            instr_and_action_ids = None
            seq_len = self.window_size
            if isinstance(input_ids, list) and isinstance(input_ids[0], str):
                # not tokenized, for speed up qwen/internvl preprocess
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

        else:
            input_ids = pad_sequences(input_ids, self.pad_token_id)
            labels = pad_sequences(labels, IGNORE_INDEX)
            attention_mask = pad_sequences(attention_mask, False)
            instr_and_action_ids = input_ids

        image_tensors = torch.stack(image_tensors)
        gripper_image_tensors = (torch.stack(gripper_image_tensors) if gripper_image_tensors[0] is not None else None)
        image_chunk = torch.stack(image_chunk) if image_chunk[0] is not None else None
        image_chunk_mask = (torch.stack(image_chunk_mask) if image_chunk_mask[0] is not None else None)
        gripper_image_chunk = (torch.stack(gripper_image_chunk) if gripper_image_chunk[0] is not None else None)
        action_tensors = torch.stack(action_tensors)
        action_mask = torch.stack(action_mask)

        action_chunk = torch.stack(action_chunk)
        action_chunk_mask = torch.stack(action_chunk_mask)

        output = {
            "rgb": image_tensors,
            "hand_rgb": gripper_image_tensors,
            "fwd_rgb_chunck": image_chunk,
            "fwd_hand_rgb_chunck": gripper_image_chunk,
            "fwd_mask": image_chunk_mask,
            "text": input_ids,
            "text_mask": attention_mask,
            "action": action_tensors,
            "action_mask": action_mask,
            "action_chunck": action_chunk,
            "chunck_mask": action_chunk_mask,
            "instr_and_action_ids": instr_and_action_ids,
            "instr_and_action_labels": labels,
            "instr_and_action_mask": attention_mask,
        }
        return output


class ActionPredictionDataset(BaseTaskDataset):
    """
    Abstract dataset base class.

    Args:
        num_workers: Number of dataloading workers for this dataset.
        batch_size: Batch size.
    """

    def __init__(
        self,
        model_name: str = "flamingo",
        mode: Literal["train", "inference"] = "train",
        organize_type: Literal["interleave", "segment"] = "interleave",
        discrete: bool = True,
        action_history: bool = True,
        image_history: bool = True,
        predict_stop_token: bool = True,
        special_history_id: int = IGNORE_INDEX,
        window_size: int = 16,
        fwd_pred_next_n: int = 2,
        n_bin=256,
        min_action=-1,
        max_action=1,
        norm_action: bool = False,
        norm_min: int = -1,
        norm_max: int = 1,
        regular_action: bool = False,
        x_mean: int = 0,
        x_std: int = 1,
        use_mu_law: bool = False,
        **kwargs,
    ):
        """
        Args:
            model_name: this value will use to build different prompt builder for different model, it will pass to get_prompt_builder function
            mode: the mode of this dataset, "train" or "inference", it will cause different data flow
            organize_type: the type you organize your output data, if you set interleave, it will be [batch size, window size, language token length + action token length(optional)],
                           else it will be [batch size, history image token length + language token length + history action token length(optional) + next action token length]
            discrete: set True if you want discrete the action to language token space
            action_history: only valid when the organize_type='segment', and if you set it False, you output data will not contain history action token
            image_history: only valid when the organize_type='segment', and if you set it False, you output data will only contain one image, else the image number will equal to window size
            predict_stop_token: only valid when the discrete=True, set True if you want the model to predict the <eos> token
            special_history_id: only valid when discrete=False and organize_type=segment, it will be the placement of the action embeding in the instr_and_action_ids

            window_size: the history length of the image / action
            fwd_pred_next_n: we need to predict fwd_pred_next_n images / actions

            n_bin: How many bins is the interval of action divided into
            min_action: the min action numerical value, if any action is lower than this, we will set these action to min_action
            max_action: the max action numerical value.

            norm_action: set True if you want to normalize the action space
            norm_min: the min action value in normalize space
            norm_max: the max action value in normalize space
            regular_action: set True if you want to regularize the action space
            x_mean: the mean action value of regular action space
            x_std: the std value of regular action space
            use_mu_law: set True if you want to use mu_law
        """
        (
            self.model_name,
            self.mode,
            self.organize_type,
            self.discrete,
            self.image_history,
            self.action_history,
            self.predict_stop_token,
            self.special_history_id,
        ) = (
            model_name,
            mode,
            organize_type,
            discrete,
            image_history,
            action_history,
            predict_stop_token,
            special_history_id,
        )

        self.window_size, self.fwd_pred_next_n = window_size, fwd_pred_next_n

        (
            self.norm_action,
            self.norm_min,
            self.norm_max,
            self.regular_action,
            self.x_mean,
            self.x_std,
            self.use_mu_law,
        ) = (norm_action, norm_min, norm_max, regular_action, x_mean, x_std, use_mu_law)

        self.n_bin, self.min_action, self.max_action = n_bin, min_action, max_action

        kwargs["task_type"] = "action"
        super().__init__(model_name=model_name, **kwargs)

    def init_batch_transform(self):
        self.action_tokenizer = None

        return ActionPredictionBatchTransform(
            action_tokenizer=self.action_tokenizer,
            special_history_id=self.special_history_id,
            model_name=self.model_name,
            tokenizer=self.tokenizer,
            text_fn=self.text_fn,
            image_fn=self.image_fn,
            window_size=self.window_size,
            fwd_pred_next_n=self.fwd_pred_next_n,
            predict_stop_token=self.predict_stop_token,
            organize_type=self.organize_type,
            discrete=self.discrete,
            image_history=self.image_history,
            action_history=self.action_history,
            mode=self.mode,
            norm_action=self.norm_action,
            norm_min=self.norm_min,
            norm_max=self.norm_max,
            x_mean=self.x_mean,
            x_std=self.x_std,
            regular_action=self.regular_action,
            use_mu_law=self.use_mu_law,
            min_action=self.min_action,
            max_action=self.max_action,
        )

    def init_collater_fn(self):
        # use or to avoid the attr exists but the value is None
        pad_token_id = getattr(self.tokenizer, "pad_token_id", 0) or 0
        return ActionPredictionPaddedCollator(
            pad_token_id=pad_token_id,
            window_size=self.window_size,
            fwd_pred_next_n=self.fwd_pred_next_n,
            discrete=self.discrete,
            organize_type=self.organize_type,
            text_fn=self.text_fn,
            model=self.model if hasattr(self, "model") else None)
