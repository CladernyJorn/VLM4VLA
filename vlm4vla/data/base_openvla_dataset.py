"""
materialize.py

Factory class for initializing Open-X RLDS-backed datasets, given specified data mixture parameters; provides and
exports individual functions for clear control flow.
"""

from functools import partial
from pathlib import Path
from typing import Dict, Literal, Any
import itertools
import torch.distributed as dist
import torch
from torch.utils.data import IterableDataset


class _RLDSDatasetByRank(IterableDataset):

    def __init__(self, origin_dataset, rank, world_size):
        super().__init__()
        self.origin_dataset = origin_dataset
        self.rank = rank
        self.world_size = world_size
        self.iterator_by_rank = itertools.islice(origin_dataset.__iter__(), rank, None, world_size)

    def __iter__(self):
        for item in self.iterator_by_rank:
            yield item

    def __len__(self):
        # drop last
        return len(self.origin_dataset) // self.world_size


class RLDSDataset(IterableDataset):

    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        image_size: int,
        chunk_action: bool = True,
        frame_num: int = -1,
        left_pad: bool = False,
        window_sample: Literal["sliding", "range"] = "sliding",
        window_size: int = 1,
        fwd_pred_next_n: int = 1,
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
        filter_langs=False,
        **kwargs,
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        from prismatic.vla.datasets.rlds.oxe import (
            OXE_NAMED_MIXTURES,
            get_oxe_dataset_kwargs_and_weights,
        )
        from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType

        super().__init__()
        self.data_root_dir, self.data_mix = data_root_dir, data_mix
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        # fmt: off
        if "aloha" in self.data_mix or "behavior" in self.data_mix:
            print("loading three camera views")
            action_proprio_normalization_type=NormalizationType.BOUNDS
        else:
            action_proprio_normalization_type=NormalizationType.BOUNDS_Q99

        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=("primary",),
            load_depth=False,
            load_proprio=False,
            load_language=True,
            action_proprio_normalization_type=action_proprio_normalization_type,
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=window_size,                                      # If we wanted to feed / predict more than one step
                chunk_action=chunk_action,
                frame_num=frame_num,
                future_action_window_size=fwd_pred_next_n,                        # For action chunking
                left_pad=left_pad,
                window_sample=window_sample,
                skip_unlabeled=True,                                # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",                 # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=(image_size, image_size),
                num_parallel_calls=16,                          # For CPU-intensive ops (decoding, resizing, etc.)
                # num_parallel_calls=100
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            # traj_transform_threads=10,
            # traj_read_threads=10,
            train=train,
            filter_langs=filter_langs
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        self.rlds_config=rlds_config
        # self.dataset=None
        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)


    def filter_dataset(self):
        pass

    def make_dataset(self, rlds_config):
        from prismatic.vla.datasets.rlds import make_interleaved_dataset

        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:

        # 获取当前进程的rank和world_size（用于DDP和多进程/多worker）
        # rank, world_size = get_rank_and_world_size()
        # # DataLoader worker信息
        # worker_info = torch.utils.data.get_worker_info()
        # if worker_info is not None:
        #     # 多worker时，每个worker分配不同的rank
        #     worker_id = worker_info.id
        #     num_workers = worker_info.num_workers
        #     # 计算全局rank和总worker数
        #     global_rank = rank * num_workers + worker_id
        #     global_world_size = world_size * num_workers

        # else:
        #     global_rank = rank
        #     global_world_size = world_size
        # print(f"rank: {rank}, world_size: {world_size}, global_rank: {global_rank}, global_world_size: {global_world_size}")
        # print("worker_info: ", worker_info)
        # 通过split_by_rank确保每个worker/进程只处理自己负责的数据
        # if hasattr(dataset, "split_by_rank"):
        #     dataset = dataset.split_by_rank(global_world_size, global_rank)
        # elif hasattr(dataset, "shard"):
        #     # 兼容TFDS Dataset的shard方法
        #     dataset = dataset.shard(num_shards=global_world_size, index=global_rank)
        self.dataset=self.dataset.as_numpy_iterator()
        # self.dataset=self.split_by_rank(global_world_size, global_rank)
        # follow spatial
        for rlds_batch in self.dataset:
            yield rlds_batch

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError(
            "IterableDataset does not implement map-style __getitem__; see __iter__ instead!"
        )

    def split_by_rank(self, world_size, rank):
        return _RLDSDatasetByRank(self, rank, world_size)


def get_rank_and_world_size():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    else:
        return 0, 1                     # 单机单进程

def count_dataset_language(dataset):
    lang_tab = {}
    ix = 0
    for data in dataset.dataset:
        lang = data["task"]["language_instruction"].numpy().decode()
        lang_id = "_".join(lang.split())
        lang_tab[lang_id] = lang_tab.get(lang_id, 0) + 1
        # if ix > 1000:
        #     break
        if ix % 100 == 0:
            print(ix)
        ix += 1
    print(lang_tab)
    return lang_tab
