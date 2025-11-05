import copy
from copy import deepcopy
import os

import lightning.pytorch as pl
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import SequentialSampler, RandomSampler

import vlm4vla
from vlm4vla.utils.dist_train import get_rank, is_dist
from vlm4vla.utils.common import collate_with_none


class GRDataModule(pl.LightningDataModule):

    def __init__(
        self,
        train_dataset,
        val_dataset,
        batch_size,
        num_workers,
        data_root="",
        **kwargs,
    ):
        super().__init__()
        self.train_dataset_config = train_dataset
        self.val_dataset_config = val_dataset
        self._train_datasets = []
        self._val_datasets = []
        self._train_loader = None
        self._val_loader = None
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.kwargs = kwargs

    def _check_data_path(self, data_cfg):
        print(self.data_root)
        if data_cfg["type"] == "ConcatDataset":
            data_cfg["datasets"] = [self._check_data_path(d) for d in data_cfg["datasets"]]
        elif "data_dir" in data_cfg and not os.path.isabs(data_cfg["data_dir"]):
            data_cfg["data_dir"] = os.path.join(self.data_root, data_cfg["data_dir"])
        return data_cfg

    def _init_dataset(self, dataset_config, batch_size, num_workers, is_training=True):
        dataset_config = self._check_data_path(dataset_config)

        # avoid modification of the self attributes
        dataset_config = copy.deepcopy(dataset_config)
        dataset_type = dataset_config.pop("type")
        # assert dataset_type in {'ConcatDataset', 'GRDataset', 'DiskCalvinDataset', 'DiskCalvinVideoDataset', 'Real_Dataset', 'VideoLLaVADataset'}
        dataset_config["is_training"] = is_training

        dataset_config.update(self.kwargs)

        # mode = dataset_config['data_dir'].split('/')[-1]
        # with open(f'dataset-{mode}.pkl', 'wb') as file:
        #     import pickle as pkl
        #     pkl.dump(dataset_config, file)

        dataset = getattr(vlm4vla.data, dataset_type)(**dataset_config)

        if is_dist():
            # default to be distributed sampler
            sampler = DistributedSampler(
                dataset,
                shuffle=True,
                drop_last=False,
                seed=self.kwargs.get("seed", 1126),
            )
        elif is_training:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
            drop_last=True,
            collate_fn=(dataset.collater if hasattr(dataset, "collater") else collate_with_none),
            prefetch_factor=2,
            pin_memory=True,
        )

        return dataset, data_loader

    def _init_iterable_dataset(self, dataset_config, batch_size, num_workers, is_training=True):
        dataset_config = self._check_data_path(dataset_config)

        # avoid modification of the self attributes
        dataset_config = copy.deepcopy(dataset_config)

        datset_type = dataset_config.pop("type")
        # assert datset_type in {'ImageTextDataset', 'RTXDataset', 'VideoLLaVADataset'}

        dataset_config.update(self.kwargs)
        dataset_config["is_training"] = is_training

        dataset = getattr(vlm4vla.data, datset_type)(**dataset_config)

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            # num_workers=4,
            num_workers=1,  # follow spatialvla
            drop_last=True,
            collate_fn=(dataset.collater if hasattr(dataset, "collater") else collate_with_none),
            pin_memory=True)

        return dataset, data_loader

    def _init_datasets(self, dataset_config, is_training, batch_size, num_workers):
        if get_rank() == 0:
            print("=" * 40)
            print("Initializing dataloader from config:")
            for k in dataset_config:
                print(f"{k}: {dataset_config[k]}")
            print(f"is_training: {is_training}")
            print(f"batch_size: {batch_size}")
            print(f"num_workers: {num_workers}")
        dataset_type = dataset_config["type"]
        assert isinstance(batch_size, int)
        assert isinstance(num_workers, int)
        if "OpenVLA" in dataset_type:
            return self._init_iterable_dataset(
                dataset_config,
                is_training=is_training,
                batch_size=batch_size,
                num_workers=num_workers,
            )
        else:
            return self._init_dataset(
                dataset_config,
                is_training=is_training,
                batch_size=batch_size,
                num_workers=num_workers,
            )

    def _init_dataset_params(self, is_training, param_name="batch_size"):
        param = getattr(self, param_name)
        if not is_training:
            # setting for val datasets
            if isinstance(param, (tuple, list)):
                if isinstance(self.val_dataset_config, (tuple, list)):
                    param = [param[0]] * len(self.val_dataset_config)
                else:
                    param = param[0]
            else:
                if isinstance(self.val_dataset_config, (tuple, list)):
                    param = [param] * len(self.val_dataset_config)
                else:
                    param = param
        else:
            if isinstance(param, int):
                if isinstance(self.train_dataset_config, (tuple, list)):
                    param = [param] * len(self.train_dataset_config)
            elif isinstance(param, (tuple, list)):
                assert isinstance(self.train_dataset_config,
                                  (tuple, list)) and len(self.train_dataset_config) == len(param)
        return param

    def initialize(self, mode="train"):
        if mode == "train":
            batch_size = self._init_dataset_params(True, "batch_size")
            num_workers = self._init_dataset_params(True, "num_workers")
            self._train_datasets, self._train_loader = self._init_datasets(self.train_dataset_config, True, batch_size,
                                                                           num_workers)

        elif mode == "val":
            batch_size = self._init_dataset_params(False, "batch_size")
            num_workers = self._init_dataset_params(False, "num_workers")
            self._val_datasets, self._val_loader = self._init_datasets(self.val_dataset_config, False, batch_size,
                                                                       num_workers)
            # if get_rank() == 0:
            #     print(f"val_loader size: {len(self._val_loader)}")

    def train_datasets(self):
        return self._train_datasets

    def val_datasets(self):
        return self._val_datasets

    def train_dataloader(self):
        self.initialize("train")
        return self._train_loader

    def val_dataloader(self):
        self.initialize("val")
        return self._val_loader
