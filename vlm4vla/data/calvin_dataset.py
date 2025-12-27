import io
import json
import logging
import os
import random
import tarfile
from dataclasses import dataclass
from multiprocessing import Value
import numpy as np
from PIL import Image

import vlm4vla
from vlm4vla.data.data_utils import (
    get_text_function,
    mu_law_companding,
    normalize_action,
    regularize_action,
)

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

try:
    from calvin_agent.datasets.utils.episode_utils import (
        get_state_info_dict,
        process_actions,
        process_depth,
        process_state,
    )
    from calvin_agent.datasets.utils.episode_utils import lookup_naming_pattern

    # import pyhash
    import torch
    from torch.utils.data import Dataset
    from vlm4vla.data.data_utils import world_to_tcp_frame

    # hasher = pyhash.fnv1_32()
    logger = logging.getLogger(__name__)
    pass
except:
    pass

Image.MAX_IMAGE_PIXELS = 1000000000
MAX_NUM_TOKENS = 256
MAX_NUM_IMAGES = 5
TINY_IMAGE_SIZE_THRESHOLD = 1
N_CHANNELS = 3
INTERLEAVED_IMAGE_SIZE = 224

MIN_KB = 10
MAX_NUM_IMAGES = 5

import logging
from pathlib import Path
from typing import Dict, Tuple, Union
from omegaconf import DictConfig

obs_config = DictConfig({
    "rgb_obs": ["rgb_static", "rgb_gripper"],
    "depth_obs": [],
    "state_obs": ["robot_obs"],
    "actions": ["rel_actions"],
    "language": ["language"],
})

prop_state = DictConfig({
    "n_state_obs": 15,
    "keep_indices": [[0, 15]],
    "robot_orientation_idx": [3, 6],
    "normalize": True,
    "normalize_robot_orientation": True,
})

from typing import Any, Dict, List, Tuple, Callable, Optional
from itertools import chain

import pickle
import torch.nn as nn
import torch.nn.functional as F


class RandomShiftsAug(nn.Module):

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
        type_x = x.dtype
        n, t, c, h, w = x.size()
        x = x.view(n * t, *x.shape[2:])
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=type_x)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        base_grid = base_grid.unsqueeze(1).repeat(1, t, 1, 1, 1)
        base_grid = base_grid.view(n * t, *base_grid.shape[2:])
        shift = torch.randint(1, 2 * self.pad + 1, size=(n * t, 1, 1, 2), device=x.device, dtype=type_x)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        x = F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)
        x = x.view(n, t, *x.shape[1:])
        return x


class BaseCalvinDataset(Dataset):
    """
    Abstract dataset base class.

    Args:
        datasets_dir: Path of folder containing episode files (string must contain 'validation' or 'training').
        obs_space: DictConfig of observation space.
        proprio_state: DictConfig with shape of prioprioceptive state.
        key: 'vis' or 'lang'.
        lang_folder: Name of the subdirectory of the dataset containing the language annotations.
        num_workers: Number of dataloading workers for this dataset.
        transforms: Dict with pytorch data transforms.
        batch_size: Batch size.
        aux_lang_loss_window: How many sliding windows to consider for auxiliary language losses, counted from the end
            of an annotated language episode.
        # TODO act_step actually is fwd_pred_next_n but not be rightly forward
    """

    def __init__(
        self,
        data_dir: Path,
        proprio_state: DictConfig = prop_state,
        lang_folder: str = "lang_annotations",
        num_workers: int = 0,
        key: str = "lang",
        obs_space: DictConfig = obs_config,
        transforms: Dict = {},
        batch_size: int = 32,
        window_size: int = 16,
        pad: bool = True,
        aux_lang_loss_window: int = 1,
        rgb_pad=-1,
        gripper_pad=-1,
        traj_cons=True,
        text_aug=False,
        dif_ws=False,
        fwd_pred_next_n=1,
        norm_action=False,
        norm_min=-1,
        norm_max=1,
        regular_action=False,
        x_mean=0,
        x_std=1,
        image_fn=None,
        episode_lookup=None,
        lang_lookup=None,
        **kwargs,
    ):
        self.observation_space = obs_space
        self.proprio_state = proprio_state
        self.transforms = transforms
        # print("transforms", self.transforms) # {}

        self.image_fn = image_fn
        self.episode_lookup = episode_lookup
        self.lang_lookup = lang_lookup

        self.with_lang = key == "lang"
        self.relative_actions = "rel_actions" in self.observation_space["actions"]
        self.pad = pad
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.window_size = window_size

        # you need to add one at act step for geting one more image than action
        self.act_step = fwd_pred_next_n + 1
        self.fwd_pred_next_n = fwd_pred_next_n

        self.norm_action = norm_action
        self.norm_min = norm_min
        self.norm_max = norm_max
        self.regular_action = regular_action
        self.x_mean = x_mean
        self.x_std = x_std
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
        # print(data_dir)
        self.abs_datasets_dir = data_dir
        if "calvin_data_copy" in str(self.abs_datasets_dir):
            lang_folder = "lang_annotations_test"
        self.lang_folder = lang_folder  # if self.with_lang else None
        self.aux_lang_loss_window = aux_lang_loss_window
        self.traj_cons = traj_cons

        self.text_aug = text_aug

        self.rgb_pad = rgb_pad
        if self.rgb_pad != -1:
            self.rgb_shift = RandomShiftsAug(rgb_pad)
        self.gripper_pad = gripper_pad
        if self.gripper_pad != -1:
            self.gripper_shift = RandomShiftsAug(gripper_pad)

        assert ("validation" in self.abs_datasets_dir.as_posix() or "training" in self.abs_datasets_dir.as_posix())
        self.validation = "validation" in self.abs_datasets_dir.as_posix()
        assert self.abs_datasets_dir.is_dir()
        logger.info(f"loading dataset at {self.abs_datasets_dir}")
        logger.info("finished loading dataset")

    def process_rgb(
        self,
        episode: Dict[str, np.ndarray],
        observation_space: DictConfig,
        transforms: Dict,
        seq_idx: int = 0,
        window_size: int = 0,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        rgb_obs_keys = observation_space["rgb_obs"]
        seq_rgb_obs_dict = {}
        for _, rgb_obs_key in enumerate(rgb_obs_keys):
            rgb_obs = episode[rgb_obs_key]
            # expand dims for single environment obs
            if len(rgb_obs.shape) != 4:
                rgb_obs = np.expand_dims(rgb_obs, axis=0)
            assert len(rgb_obs.shape) == 4
            if window_size == 0 and seq_idx == 0:  # single file loader
                # To Square image
                seq_rgb_obs_ = torch.from_numpy(rgb_obs).byte()
            else:  # episode loader
                seq_rgb_obs_ = torch.from_numpy(rgb_obs[seq_idx:seq_idx + window_size]).byte()

            if rgb_obs_key in transforms:
                seq_rgb_obs_ = transforms[rgb_obs_key](seq_rgb_obs_)
            seq_rgb_obs_dict[rgb_obs_key] = seq_rgb_obs_
        # shape: N_rgb_obs x (BxHxWxC)
        return {"rgb_obs": seq_rgb_obs_dict}

    def process_language(self, episode: Dict[str, np.ndarray], transforms: Dict, with_lang: bool):
        if with_lang:
            return {"lang": episode["language"]}
        else:
            return {"lang": "execute random action."}

    def discretize_action_bins(self, action, action_bin=256):
        action_min = -1.001
        action_max = 1.001
        action_len = (action_max - action_min) / action_bin
        action = torch.FloatTensor(action)
        # pose_action = (pose_action - action_min) / action_len # original wrong in vlm4vla?
        pose_action = (action - action_min) / action_len
        pose_action = torch.floor(pose_action).long().view(-1).tolist()
        pose_action[-1] = int(action[-1])
        return pose_action

    def process_rt2_ag_text(self, text, action):
        action_id = self.discretize_action_bins(action)
        action_text = ["<Action_{}>".format(i) for i in action_id]
        action_text.append("<Gripper_{}>".format(action[-1]))

        return action_text

    def __getitem__(self, idx: Union[int, Tuple[int, int]], fixed_seed=False) -> Dict:
        """
        Get sequence of dataset.

        Args:
            idx: Index of the sequence.

        Returns:
            Loaded sequence.
        """
        head = False
        sequence = self._get_sequences(idx, self.window_size, head=head)
        import copy

        new_list = []
        np_rgb = copy.deepcopy(sequence["rgb_obs"]["rgb_static"].numpy())
        for i in range(np_rgb.shape[0]):
            new_list.append(Image.fromarray(np_rgb[i, :, :, :].astype(np.uint8)))
        assert self.image_fn is not None
        image_tensors = self.image_fn(new_list)  # 11264,1176
        if self.rgb_pad != -1:
            if self.traj_cons:
                image_tensors = self.rgb_shift.forward_traj(image_tensors.unsqueeze(0)).squeeze(0)
            else:
                image_tensors = self.rgb_shift(image_tensors)

        sequence["rgb_obs"]["rgb_static"] = image_tensors
        new_list = []
        np_gripper = copy.deepcopy(sequence["rgb_obs"]["rgb_gripper"].numpy())
        for i in range(np_gripper.shape[0]):
            new_list.append(Image.fromarray(np_gripper[i, :, :, :].astype(np.uint8)))

        gripper_tensors = self.image_fn(new_list)
        if self.gripper_pad != -1:
            if self.traj_cons:
                gripper_tensors = self.gripper_shift.forward_traj(gripper_tensors.unsqueeze(0)).squeeze(0)
            else:
                gripper_tensors = self.gripper_shift(gripper_tensors)

        sequence["rgb_obs"]["rgb_gripper"] = gripper_tensors
        # print(pad_size, len(new_list))
        return sequence

    def _get_sequences(self, idx: Union[int, Tuple[int, int]], window_size: int, head: bool = False) -> Dict:
        """
        Load sequence of length window_size.

        Args:
            idx: Index of starting frame.
            window_size: Length of sampled episode.

        Returns:
            dict: Dictionary of tensors of loaded sequence with different input modalities and actions.
        """

        episode = self._load_episode(idx, window_size)

        seq_state_obs = process_state(episode, self.observation_space, self.transforms, self.proprio_state)
        seq_rgb_obs = self.process_rgb(episode, self.observation_space, self.transforms)
        seq_depth_obs = process_depth(episode, self.observation_space, self.transforms)
        seq_acts = process_actions(episode, self.observation_space, self.transforms)

        info = get_state_info_dict(episode)
        seq_lang = self.process_language(episode, self.transforms, self.with_lang)
        info = self._add_language_info(info, idx)
        seq_dict = {
            **seq_state_obs,
            **seq_rgb_obs,
            **seq_depth_obs,
            **seq_acts,
            **info,
            **seq_lang,
        }  # type:ignore
        seq_dict["idx"] = idx  # type:ignore
        seq_dict["action_mask"] = episode["action_mask"]
        seq_dict["image_mask"] = episode["image_mask"]
        return seq_dict

    def _load_episode(self, idx: Union[int, Tuple[int, int]], window_size: int) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def __len__(self) -> int:
        """
        Returns:
            Size of the dataset.
        """
        assert self.episode_lookup is not None
        return len(self.episode_lookup)

    def _pad_sequence(self, seq: Dict, pad_size: int, head: bool = False) -> Dict:
        """
        Pad a sequence by repeating the last frame.

        Args:
            seq: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded sequence.
        """
        seq.update({"robot_obs": self._pad_with_repetition(seq["robot_obs"], pad_size)})
        seq.update({"rgb_obs": {k: self._pad_with_repetition(v, pad_size, head) for k, v in seq["rgb_obs"].items()}})
        seq.update({"depth_obs": {k: self._pad_with_repetition(v, pad_size, head) for k, v in seq["depth_obs"].items()}})
        #  todo: find better way of distinguishing rk and play action spaces
        if not self.relative_actions:
            if head:
                seq_acts = self._pad_with_zeros(seq["actions"], pad_size, head)
            else:
                # repeat action for world coordinates action space
                seq.update({"actions": self._pad_with_repetition(seq["actions"], pad_size, head)})
        else:
            # for relative actions zero pad all but the last action dims and repeat last action dim (gripper action)
            if head:
                seq_acts = self._pad_with_zeros(seq["actions"], pad_size, head)
            else:
                seq_acts = torch.cat(
                    [
                        self._pad_with_zeros(seq["actions"][..., :-1], pad_size, head),
                        self._pad_with_repetition(seq["actions"][..., -1:], pad_size, head),
                    ],
                    dim=-1,
                )
            seq.update({"actions": seq_acts})
        seq.update(
            {"state_info": {
                k: self._pad_with_repetition(v, pad_size, head) for k, v in seq["state_info"].items()
            }})
        return seq

    @staticmethod
    def _pad_with_repetition(input_tensor: torch.Tensor, pad_size: int, head: bool = False) -> torch.Tensor:
        """
        Pad a sequence Tensor by repeating last element pad_size times.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        if head:
            last_repeated = torch.repeat_interleave(torch.unsqueeze(input_tensor[0], dim=0), repeats=pad_size, dim=0)
            padded = torch.vstack((last_repeated, input_tensor))
        else:
            last_repeated = torch.repeat_interleave(torch.unsqueeze(input_tensor[-1], dim=0), repeats=pad_size, dim=0)
            padded = torch.vstack((input_tensor, last_repeated))
        return padded

    @staticmethod
    def _pad_with_zeros(input_tensor: torch.Tensor, pad_size: int, head: bool = False) -> torch.Tensor:
        """
        Pad a Tensor with zeros.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        zeros_repeated = torch.repeat_interleave(
            torch.unsqueeze(torch.zeros(input_tensor.shape[-1]), dim=0),
            repeats=pad_size,
            dim=0,
        )
        if head:
            padded = torch.vstack((zeros_repeated, input_tensor))
        else:
            padded = torch.vstack((input_tensor, zeros_repeated))
        return padded

    def _add_language_info(self, info: Dict, idx: Union[int, Tuple[int, int]]) -> Dict:
        """
        If dataset contains language, add info to determine if this sequence will be used for the auxiliary losses.

        Args:
            info: Info dictionary.
            idx: Sequence index.

        Returns:
            Info dictionary with updated information.
        """
        if not self.with_lang:
            return info
        assert self.lang_lookup is not None
        assert isinstance(idx, int)
        use_for_aux_lang_loss = (
            idx + self.aux_lang_loss_window >= len(self.lang_lookup) or
            self.lang_lookup[idx] < self.lang_lookup[idx + self.aux_lang_loss_window])
        info["use_for_aux_lang_loss"] = use_for_aux_lang_loss
        return info


class DiskCalvinDataset(BaseCalvinDataset):
    """
    Dataset that loads episodes as individual files from disk.
    Args:
        skip_frames: Skip this amount of windows for language dataset.
        save_format: File format in datasets_dir (pkl or npz).
        pretrain: Set to True when pretraining.
    """

    def __init__(
        self,
        image_fn: Callable,  # image_process
        tokenizer: Callable,  # model.model.tokenizer ( backbone.tokenizer )
        *args: Any,
        skip_frames: int = 1,  # 1, not input
        save_format: str = "npz",  # npz, not input
        pretrain: bool = False,  # False, not input
        partial_data=False,  # False, not input
        decoder_type="lstm",  # lstm, not input, not used
        discrete_action=False,  # False
        action_tokenizer=None,
        model_name="vicuna",  # config.model
        predict_stop_token=True,
        use_mu_law=False,  # False, not in config
        mu_val=255,  # 255, not in config
        n_bin=256,  # 256, not in config
        min_action=-1,  # -1, not in config
        max_action=1,  # 1, not in config
        task_type="calvin_action",  # calvin_action, not in config
        tcp_rel=False,  # False, not in config
        few_shot=False,  # False, not in config
        exclude_tasks=[],  # [], not in config
        **kwargs:
        Any,  # tokenizer_conifg, fwd_pred_next_n, window_size, image_size, discrete, discrete_action_history, act_step=10, norm_action=True, norm_min=-0.65, norm_max=0.65, regular_action=False, x_mean=0, x_std=1, weights=None
    ):
        super().__init__(*args, **kwargs)
        self.decoder_type = decoder_type
        self.save_format = save_format
        self.image_fn = image_fn

        self.tokenizer = tokenizer
        self.text_fn = get_text_function(self.tokenizer, model_name)
        self.partial_data = partial_data
        if self.save_format == "pkl":
            self.load_file = load_pkl
        elif self.save_format == "npz":
            self.load_file = load_npz
        else:
            raise NotImplementedError
        self.pretrain = pretrain
        self.skip_frames = skip_frames
        self.use_mu_law = use_mu_law
        self.mu_val = mu_val
        self.task_type = task_type
        self.tcp_rel = tcp_rel
        self.few_shot = few_shot
        self.exclude_tasks = exclude_tasks
        # print(self.task_type)  # calvin_action

        self.naming_pattern, self.n_digits = lookup_naming_pattern(self.abs_datasets_dir, self.save_format)
        (
            self.episode_lookup,
            self.lang_lookup,
            self.right_pad_lookup,
            self.lang_ann,
            self.lang_task,
        ) = self._build_file_indices_lang(self.abs_datasets_dir)

        self.model_name = model_name
        self.discrete_action = discrete_action
        self.predict_stop_token = predict_stop_token
        assert self.discrete_action is False  # not using discrete action

    def _get_episode_name(self, file_idx: int) -> Path:
        """
        Convert file idx to file path.
        Args:
            file_idx: index of starting frame.
        Returns:
            Path to file.
        """
        return Path(f"{self.naming_pattern[0]}{file_idx:0{self.n_digits}d}{self.naming_pattern[1]}")

    def _load_episode(self, idx: Union[int, Tuple[int, int]], window_size: int) -> Dict[str, np.ndarray]:
        """
        Load consecutive frames saved as individual files on disk and combine to episode dict.
        Args:
            idx: Index of first frame.
            window_size: Length of sampled episode.
        Returns:
            episode: Dict of numpy arrays containing the episode where keys are the names of modalities.
        """
        assert isinstance(idx, int)
        start_idx = self.episode_lookup[idx]

        end_idx = start_idx + window_size + self.act_step - 1
        right_pad = self.right_pad_lookup[idx]
        idx_range = np.arange(start_idx, end_idx)
        action_mask = np.ones_like(idx_range)
        image_mask = np.ones_like(idx_range)
        if right_pad != 0:
            idx_range[right_pad:] = idx_range[right_pad]
            action_mask[right_pad:] = 0
            image_mask[right_pad:] = 0

        keys = list(chain(*self.observation_space.values()))
        keys.remove("language")
        keys.append("scene_obs")
        episodes = [self.load_file(self._get_episode_name(file_idx)) for file_idx in idx_range]
        episode = {key: np.stack([ep[key] for ep in episodes]) for key in keys}
        if self.with_lang:
            episode["language"] = self.lang_ann[self.lang_lookup[idx]]
            if self.text_aug:
                task = self.lang_task[self.lang_lookup[idx]]
                # enrich_lang = random.choice(self.enrich_lang[task] +
                #                             [episode["language"]])  # original from vlm4vla, wrong?
                enrich_lang = random.choice([episode["language"]])
                episode["language"] = enrich_lang
        episode["action_mask"] = action_mask
        episode["image_mask"] = image_mask
        return episode

    def _build_file_indices_lang(self, abs_datasets_dir: Path):
        """
        This method builds the mapping from index to file_name used for loading the episodes of the language dataset.
        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.
        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
            lang_lookup: Mapping from training example to index of language instruction.
            lang_ann: Language embeddings.
        """
        assert abs_datasets_dir.is_dir()

        episode_lookup = []
        right_pad_lookup = []

        try:
            print(
                "trying to load lang data from: ",
                abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy",
            )
            lang_data = np.load(
                abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy",
                allow_pickle=True,
            ).item()
        except Exception:
            print(
                "Exception, trying to load lang data from: ",
                abs_datasets_dir / "auto_lang_ann.npy",
            )
            lang_data = np.load(abs_datasets_dir / "auto_lang_ann.npy", allow_pickle=True).item()

        ep_start_end_ids = lang_data["info"]["indx"]  # each of them are 64
        lang_ann = lang_data["language"]["ann"]  # length total number of annotations
        lang_task = lang_data["language"]["task"]
        lang_lookup = []
        # add support for partial calvin data
        # partial_st_ed_list = []
        partial_st_ed_list = load_partial_traj_data()
        few_shot_st_ed_list = load_few_shot_traj_data()
        # import pdb; pdb.set_trace()
        for i, (start_idx, end_idx) in enumerate(ep_start_end_ids):
            if self.partial_data:
                if (start_idx, end_idx) not in partial_st_ed_list:
                    continue
            if self.few_shot:
                if (start_idx, end_idx) not in few_shot_st_ed_list:
                    continue
            if lang_task[i] in self.exclude_tasks:
                continue
            cnt = 0
            right_pad = end_idx - start_idx - self.act_step - self.window_size + 1
            for idx in range(start_idx, end_idx + 1 - self.window_size):
                if cnt % self.skip_frames == 0:
                    lang_lookup.append(i)
                    episode_lookup.append(idx)
                    right_pad_lookup.append(min(0, right_pad))
                right_pad -= 1
                cnt += 1

        return (
            np.array(episode_lookup),
            lang_lookup,
            right_pad_lookup,
            lang_ann,
            lang_task,
        )

    def _build_file_indices(self, abs_datasets_dir: Path) -> Tuple[np.ndarray, List[int]]:
        """
        This method builds the mapping from index to file_name used for loading the episodes of the non language
        dataset.
        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.
        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
        """
        assert abs_datasets_dir.is_dir()

        episode_lookup = []
        right_pad_lookup = []

        ep_start_end_ids = np.load(abs_datasets_dir / "ep_start_end_ids.npy")
        logger.info(f'Found "ep_start_end_ids.npy" with {len(ep_start_end_ids)} episodes.')
        for start_idx, end_idx in ep_start_end_ids:
            right_pad = end_idx - start_idx - self.act_step - self.window_size
            for idx in range(start_idx, end_idx + 2 - self.window_size):
                episode_lookup.append(idx)
                right_pad_lookup.append(min(0, right_pad))
                right_pad -= 1
        return np.array(episode_lookup), right_pad_lookup

    # NOTE
    def collater(self, sample):
        if self.norm_action:
            new_sample = []
            for s in sample:
                s["actions"] = normalize_action(s["actions"], self.norm_min, self.norm_max, maintain_last=True)
                new_sample.append(s)
            sample = new_sample

        if self.regular_action:
            new_sample = []
            for s in sample:
                s["actions"] = regularize_action(s["actions"], self.x_mean, self.x_std)
                new_sample.append(s)
            sample = new_sample
            pass

        if self.use_mu_law:
            new_sample = []
            for s in sample:
                s["actions"] = mu_law_companding(s["actions"], self.mu_val)
                new_sample.append(s)
            sample = new_sample

        action_tensors = torch.from_numpy(np.array([np.stack(s["actions"]) for s in sample]))[:, :-1]
        # 把多读的一个1去掉（最后一个），ws和fwd_pred_next_n中的有一帧是重叠的
        # print("action_tensors after normalize: ", action_tensors)  # tcp没用，最后一位-1/1，其他位置 0.几 正负都有
        # print("action_tensors.shape after normalize: ", action_tensors.shape)  # 16,10,7
        action_mask = torch.from_numpy(np.array([np.stack(s["action_mask"]) for s in sample]))[:, :-1]
        robot_obs = torch.from_numpy(np.array([np.stack(s["robot_obs"]) for s in sample]))[:, :-1]

        if self.tcp_rel:
            action_tensors = world_to_tcp_frame(action_tensors, robot_obs)
        # print("action_tensors after tcp_rel: ", action_tensors)  # 最后一位-1/1，其他位置 0.几 正负都有
        # print("action_tensors.shape after tcp_rel: ", action_tensors.shape)  # 16,10,7

        image_mask = torch.from_numpy(np.array([np.stack(s["image_mask"]) for s in sample]))
        image_tensors = torch.stack([s["rgb_obs"]["rgb_static"] for s in sample])
        gripper_tensors = torch.stack([s["rgb_obs"]["rgb_gripper"] for s in sample])

        stacked_language = [s["lang"] for s in sample]
        text_tensors, attention_mask = self.text_fn(stacked_language)  # [input_texts],None for qwen25vl
        action_tensors[..., -1] = ((action_tensors[..., -1] + 1) // 2).float()

        image_chunk = image_tensors.unfold(1, self.fwd_pred_next_n, 1).permute(0, 1, 5, 2, 3, 4)[:, 1:]
        image_tensors = image_tensors[:, :self.window_size]
        # print("image_tensors after unfold: ", image_tensors.shape)  # 16，1，3，224，224
        if gripper_tensors is not None:
            gripper_chunk = gripper_tensors.unfold(1, self.fwd_pred_next_n, 1).permute(0, 1, 5, 2, 3, 4)[:, 1:]
            gripper_tensors = gripper_tensors[:, :self.window_size]
        else:
            gripper_chunk = None

        fwd_mask = image_mask.unfold(1, self.fwd_pred_next_n, 1)[:, 1:]

        action_chunck = action_tensors.unfold(1, self.fwd_pred_next_n, 1).permute(0, 1, 3, 2)
        action_mask = action_mask.unfold(1, self.fwd_pred_next_n, 1)
        # print("action_chunck.shape: ", action_chunck.shape)  # 16,8,10,7
        # robot_obs_chunk = robot_obs.unfold(1, self.fwd_pred_next_n, 1).permute(0, 1, 3, 2)
        # if self.tcp_rel:
        #     action_chunck = world_to_tcp_frame(action_chunck, robot_obs_chunk)

        bs = len(sample)
        instr_and_action_ids = None
        instr_and_action_labels = None
        instr_and_action_mask = None

        res = {
            "rgb": image_tensors,
            "hand_rgb": gripper_tensors,
            "action": action_tensors,
            "text": text_tensors,
            "text_mask": attention_mask,
            "fwd_rgb_chunck": image_chunk,
            "fwd_hand_rgb_chunck": gripper_chunk,
            "fwd_mask": fwd_mask,
            "action_chunck": action_chunck,  # action最后没用，用的action_chunck
            "chunck_mask": action_mask,
            "instr_and_action_ids": instr_and_action_ids,
            "instr_and_action_labels": instr_and_action_labels,
            "instr_and_action_mask": instr_and_action_mask,
            "raw_text": stacked_language,
            "data_source": self.task_type,
        }
        return res


def load_pkl(filename: Path) -> Dict[str, np.ndarray]:
    with open(filename, "rb") as f:
        return pickle.load(f)


def load_npz(filename: Path) -> Dict[str, np.ndarray]:
    return np.load(filename.as_posix())


class SharedEpoch:

    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value("i", epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: Optional[DistributedSampler] = None
    shared_epoch: Optional[SharedEpoch] = None
    dataset: Optional[Dataset] = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def preprocess_image(sample, image_processor):
    image = [image_processor(s).unsqueeze(0) for s in sample]
    image = torch.cat(image, dim=0)
    # apply random horizontal flip and color jitter
    return image


def preprocess_text_calvin(sample, tokenizer, decoder_type="lstm"):
    tokenizer.padding_side = "right"
    max_length = 48 if decoder_type == "rt2_enc" else 32
    if decoder_type == "rt2_enc":
        action_str = "".join([f"<Action_{i}>" for i in range(7)])
        sample = [(f"<image>{s.strip()}{action_str}<|endofchunk|>{tokenizer.eos_token}") for s in sample]

    else:
        sample = [(f"<image>{s.strip()}<|endofchunk|>{tokenizer.eos_token}") for s in sample]
    text = tokenizer(
        sample,
        max_length=max_length,
        padding="longest",
        truncation="only_first",
        return_tensors="pt",
    )
    return text["input_ids"], text["attention_mask"]


def load_partial_traj_data():
    file = open(
        f"{Path(os.path.abspath(vlm4vla.__path__[0])).parent.as_posix()}/configs/data/calvin/data_name_list.txt",
        "r",
    )
    lines = file.readlines()
    lines = [tuple([int(_) for _ in l.split()[1:]]) for l in lines]
    return lines


def load_few_shot_traj_data():
    file = json.load(
        open(
            f"{Path(os.path.abspath(vlm4vla.__path__[0])).parent.as_posix()}/configs/data/calvin/10_shot_task_data.json",
            "r",
        ))
    res = []
    for task in file:
        res.extend([tuple(_) for _ in file[task]])
    return res
