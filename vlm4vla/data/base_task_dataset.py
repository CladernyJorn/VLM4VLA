from typing import Dict, Callable, Union, List
from abc import ABC, abstractmethod
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import PreTrainedTokenizerBase

from vlm4vla.data.data_utils import get_text_function
from vlm4vla.utils.model_utils import build_tokenizer

IGNORE_INDEX = -100


class RandomShiftsAug(nn.Module):
    """
    Random shift one image using forward, or some images using forward_traj
    """

    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)

    def forward_traj(self, x):
        n, t, c, h, w = x.size()
        x = x.reshape(n * t, *x.shape[2:])
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        base_grid = base_grid.unsqueeze(1).repeat(1, t, 1, 1, 1)
        base_grid = base_grid.reshape(n * t, *base_grid.shape[2:])
        shift = torch.randint(1, 2 * self.pad + 1, size=(n * t, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        x = F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)
        x = x.reshape(n, t, *x.shape[1:])
        return x


class BaseTaskDataset(Dataset, ABC):

    def __init__(
        self,
        task_type: str,
        image_fn: Callable,
        tokenizer: Union[Dict, PreTrainedTokenizerBase],
        rgb_pad: int = -1,
        gripper_pad: int = -1,
        traj_cons: bool = True,
        model_name: str = "",  # added to init text_fn
        **kwargs,
    ):
        self.task_type = task_type
        self.init_tokenizer(tokenizer)
        self.init_image_fn(rgb_pad, gripper_pad, traj_cons, image_fn)
        self.batch_transform = self.init_batch_transform()
        self.collater_fn = self.init_collater_fn()
        self.model_name = model_name

    def init_tokenizer(self, tokenizer):
        if isinstance(tokenizer, dict):
            tokenizer_type = tokenizer["tokenizer_type"]
            max_text_len = tokenizer["max_text_len"]
            tokenizer = build_tokenizer(tokenizer_config=tokenizer)
            self.tokenizer = tokenizer
            self.text_fn = get_text_function(tokenizer, tokenizer_type, max_text_len)
        else:
            # print("here! tokenizer is not a dict") # default using here
            self.tokenizer = tokenizer
            # self.text_fn = tokenizer
            # self.text_fn = get_text_function(tokenizer, tokenizer_type="default")
            self.text_fn = get_text_function(self.tokenizer, self.model_name)

    def init_image_fn(self, rgb_pad, gripper_pad, traj_cons, image_fn: Callable):
        # print("here! rgb_pad, gripper_pad", rgb_pad, gripper_pad) -1,-1
        if isinstance(image_fn, tuple):
            image_fn, self.model = image_fn
            # print("here! image_fn, model", image_fn, self.model)
        self.rgb_shift = RandomShiftsAug(rgb_pad) if rgb_pad != -1 else None
        self.gripper_shift = RandomShiftsAug(gripper_pad) if gripper_pad != -1 else None

        def pad_image_fn(images: Union[np.ndarray, List[Image.Image]], static: bool = True):
            if isinstance(images, np.ndarray):
                images = [Image.fromarray(image) for image in images]
            images = image_fn(images)
            shift_model = self.rgb_shift if static else self.gripper_shift
            if shift_model is None:
                # print("here! shift_model is None")
                return images
            if traj_cons:
                return shift_model.forward_traj(images.unsqueeze(0)).squeeze(0)
            else:
                return shift_model(images)

        self.image_fn = pad_image_fn

    @abstractmethod
    def init_batch_transform(self):
        raise NotImplementedError("You need to implentment batch transform function")

    @abstractmethod
    def init_collater_fn(self):
        raise NotImplementedError("You need to implentment collater function")

    def collater(self, samples):
        return_data = self.collater_fn(samples)
        return_data["data_source"] = self.task_type
        return return_data
