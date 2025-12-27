import argparse
import os
from typing import List

import torch

from vlm4vla.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict
from vlm4vla.utils.config_utils import load_config

CPU_DEVICE = torch.device("cpu")


def parse_model_load_path_from_config(config_file):
    configs = load_config(config_file)
    return configs.get("model_load_path", None)


def parse_model_load_path_from_unknown(arg_list: List):
    for i, v in enumerate(arg_list):
        if v == "model_load_path":
            return arg_list[i + 1]
    return None


def get_converted_fp32_paths(deepspeed_ckpt_path):
    deepspeed_ckpt_path = deepspeed_ckpt_path.rstrip("/")
    ckpt_dir = os.path.dirname(deepspeed_ckpt_path)
    ckpt_name = os.path.basename(deepspeed_ckpt_path)
    fp32_ckpt_name = f"{ckpt_name}.fp32.pt"
    converted_path = os.path.join(ckpt_dir, fp32_ckpt_name)
    return converted_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="the config file for training")
    args, unknown_args = parser.parse_known_args()
    model_load_path = parse_model_load_path_from_unknown(unknown_args) or parse_model_load_path_from_config(args.config)
    if model_load_path is None or not os.path.isdir(model_load_path):
        print("No deepspeed checkpoint is needed in this training.")
        return
    converted_path = get_converted_fp32_paths(model_load_path)
    convert_zero_checkpoint_to_fp32_state_dict(model_load_path, converted_path)


if __name__ == "__main__":
    main()
