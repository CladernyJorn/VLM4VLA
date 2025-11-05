import os
import torch
import numpy as np
from lightning import seed_everything
import argparse

import numpy as np

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import tqdm
from libero.libero import benchmark
from eval.libero.model_wrapper import BaseModelInference
from eval.libero.libero_evaluator import libero_evaluator
import tensorflow as tf


def parse_range_tuple(t):
    return np.linspace(t[0], t[1], int(t[2]))


def get_args():
    # parse command-line arguments
    seed_everything(0, workers=True)  # type:ignore
    parser = argparse.ArgumentParser()

    parser.add_argument("--logging-dir", type=str, default="./results")
    parser.add_argument("--tf-memory-limit", type=int, default=3072, help="Tensorflow memory limit")
    parser.add_argument("--config_path", type=str, default=None, help="path to the config file")
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        nargs="+",
        default="",
        help="checkpoint directory of the training",
    )
    parser.add_argument(
        "--task_suite_name",
        type=str,
        default="",
        help="task suite name",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="checkpoint directory of the training",
    )
    parser.add_argument("--center_crop", default=True, type=bool)
    parser.add_argument("--num_steps_wait", default=10, type=int)
    parser.add_argument("--num_trials_per_task", default=50, type=int)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--execute_step", type=int, default=1, help="execute step for the model")
    args = parser.parse_args()
    assert args.task_suite_name in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
    return args


if __name__ == "__main__":
    from vlm4vla.utils.config_utils import load_config

    args = get_args()

    config_path = args.config_path
    ckpt_dir = args.ckpt_dir
    ckpt_idx = 0

    # Loading configs
    assert config_path != None
    configs = load_config(config_path)

    # change all path in configs to new path
    new_dir = "/home/disk1/jianke_z"
    old_dir = "/mnt/zjk/jianke_z"
    keys_to_change = [
        "model_path", "model_config", ["tokenizer", "pretrained_model_name_or_path"],
        ["vlm", "pretrained_model_name_or_path"]
        # ["train_dataset", "data_dir"], ["val_dataset", "data_dir"]
    ]

    def check_qwen_dir(path):
        if not os.path.exists(path):
            if "Qwen2.5" in path:
                has_configjson = "config.json" in path
                if "3b" in path or "3B" in path:
                    new_path = "/home/disk1/jianke_z/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/c747f21f03e7d0792c30766310bd7d8de17eeeb3"
                elif "7b" in path or "7B" in path:
                    new_path = "/home/disk1/jianke_z/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5"
                else:
                    raise ValueError(f"Unknown Qwen2.5 model size in {configs[key[0]][key[1]]}")
                if has_configjson:
                    new_path += "/config.json"
                return new_path
        else:
            return path

    for key in keys_to_change:
        if isinstance(key, list):
            configs[key[0]][key[1]] = configs[key[0]][key[1]].replace(old_dir, new_dir)
            if "dataset" in key[0]:
                configs[key[0]][key[1]] = configs[key[0]][key[1]].replace("data/robotics_0707/", "")
            configs[key[0]][key[1]] = check_qwen_dir(configs[key[0]][key[1]])
        else:
            configs[key] = configs[key].replace(old_dir, new_dir)
            configs[key] = check_qwen_dir(configs[key])

    if "pi0_cfg" in configs.keys():
        print("Changing pi0_cfg path")
        configs["pi0_cfg"] = configs["pi0_cfg"].replace(old_dir + "/vlm4vla", new_dir + "/VLM4VLA")

    args.model_name = configs["config"].split("/")[-1].split(".")[0]
    args.model_name += f'_{configs["exp_name"]}'
    os.environ["DISPLAY"] = ""
    # prevent a single jax process from taking up all the GPU memory
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 0:
        # prevent a single tf process from taking up all the GPU memory
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=args.tf_memory_limit)],
        )

    from vlm4vla.utils.eval_utils import sort_ckpt

    # print(ckpt_dir)
    if isinstance(ckpt_dir, list):
        ckpt_dir = ckpt_dir[0]
    if args.ckpt_path is None:
        ckpt_files, ckpt_steps = sort_ckpt(ckpt_dir)
        if ckpt_idx >= len(ckpt_files):
            exit(0)
        ckpt_path = ckpt_files[ckpt_idx]
        ckpt_step = ckpt_steps[ckpt_idx]
        ckpt_dir = os.path.dirname(ckpt_path)
    else:
        import copy

        ckpt_path = args.ckpt_path or copy.copy(ckpt_dir)
        ckpt_dir = os.path.dirname(ckpt_path)
        ckpt_step = 0

    # Handle DeepSpeed ckpt
    if os.path.isdir(ckpt_path):
        target_ckpt_path = ckpt_path.replace(".ckpt", ".pt")
        from vlm4vla.utils.zero_to_fp32 import (
            convert_zero_checkpoint_to_fp32_state_dict,)

        print(f"converting {ckpt_path} to {target_ckpt_path}")
        convert_zero_checkpoint_to_fp32_state_dict(ckpt_path, target_ckpt_path)
        ckpt_path = target_ckpt_path

    from vlm4vla.utils.config_utils import get_exp_name

    eval_exp_name = get_exp_name(f"{os.path.basename(config_path)}", mode="eval")
    # if args.no_cache:
    #     eval_log_dir = f"{ckpt_dir}/{ckpt_path.split('/')[-1].split('.')[0].split('=')[-1]}-eval"
    # else:
    #     eval_log_dir = os.path.join(CACHE_ROOT, eval_exp_name)
    eval_log_dir = f"{ckpt_dir}/{args.task_suite_name}_execute_step_{args.execute_step}/{ckpt_path.split('/')[-1].split('.')[0].split('=')[-1]}-eval"
    if args.center_crop:
        eval_log_dir += "-centercrop"
    args.logging_dir = eval_log_dir
    # os.system(f"mkdir {eval_log_dir}")
    os.makedirs(eval_log_dir, exist_ok=True)
    os.system(f"chmod 777 -R {eval_log_dir}")

    model = BaseModelInference(
        ckpt_path=ckpt_path,
        configs=configs,
        device=torch.device("cuda"),
        save_dir=eval_log_dir,
        execute_step=args.execute_step,
        policy_setup=args.task_suite_name,
        center_crop=args.center_crop)

    # run real-to-sim evaluation
    success_arr = libero_evaluator(model, args)
    print(args)
    print(" " * 10, "Average success", np.mean(success_arr))
    json.dump(
        {
            "success_rate": np.mean(success_arr) * 100,
            "test_num": len(success_arr),
            "success_num": int(np.sum(success_arr)),
        }, open(os.path.join(eval_log_dir, f"{args.task_suite_name}_{np.mean(success_arr):.4f}.json"), "w"))
