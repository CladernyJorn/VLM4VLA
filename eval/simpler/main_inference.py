import os
import torch
import numpy as np
import tensorflow as tf

from simpler_env.evaluation.argparse import get_args
from eval.simpler.env_utlis import DictAction
from eval.simpler.maniskill2_evaluator import maniskill2_evaluator
from eval.simpler.model_wrapper import BaseModelInference

import argparse

import numpy as np
from sapien.core import Pose
from transforms3d.euler import euler2quat
import json


def parse_range_tuple(t):
    return np.linspace(t[0], t[1], int(t[2]))


def get_args():
    # parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy-model",
        type=str,
        default="rt1",
        help="Policy model type; e.g., 'rt1', 'octo-base', 'octo-small'",
    )
    parser.add_argument(
        "--policy-setup",
        type=str,
        default="google_robot",
        help="Policy model setup; e.g., 'google_robot', 'widowx_bridge'",
    )
    parser.add_argument("--ckpt-path", type=str, default=None)
    parser.add_argument("--env-name", type=str, required=True)
    parser.add_argument(
        "--additional-env-save-tags",
        type=str,
        default=None,
        help="Additional tags to save the environment eval results",
    )
    parser.add_argument("--scene-name", type=str, default="google_pick_coke_can_1_v4")
    parser.add_argument("--enable-raytracing", action="store_true")
    parser.add_argument("--robot", type=str, default="google_robot_static")
    parser.add_argument(
        "--obs-camera-name",
        type=str,
        default=None,
        help="Obtain image observation from this camera for policy input. None = default",
    )
    parser.add_argument("--action-scale", type=float, default=1.0)

    parser.add_argument("--control-freq", type=int, default=3)
    parser.add_argument("--sim-freq", type=int, default=513)
    parser.add_argument("--max-episode-steps", type=int, default=80)
    parser.add_argument("--rgb-overlay-path", type=str, default=None)
    parser.add_argument(
        "--robot-init-x-range",
        type=float,
        nargs=3,
        default=[0.35, 0.35, 1],
        help="[xmin, xmax, num]",
    )
    parser.add_argument(
        "--robot-init-y-range",
        type=float,
        nargs=3,
        default=[0.20, 0.20, 1],
        help="[ymin, ymax, num]",
    )
    parser.add_argument(
        "--robot-init-rot-quat-center",
        type=float,
        nargs=4,
        default=[1, 0, 0, 0],
        help="[x, y, z, w]",
    )
    parser.add_argument(
        "--robot-init-rot-rpy-range",
        type=float,
        nargs=9,
        default=[0, 0, 1, 0, 0, 1, 0, 0, 1],
        help="[rmin, rmax, rnum, pmin, pmax, pnum, ymin, ymax, ynum]",
    )
    parser.add_argument(
        "--obj-variation-mode",
        type=str,
        default="xy",
        choices=["xy", "episode"],
        help="Whether to vary the xy position of a single object, or to vary predetermined episodes",
    )
    parser.add_argument("--obj-episode-range", type=int, nargs=2, default=[0, 60], help="[start, end]")
    parser.add_argument(
        "--obj-init-x-range",
        type=float,
        nargs=3,
        default=[-0.35, -0.12, 5],
        help="[xmin, xmax, num]",
    )
    parser.add_argument(
        "--obj-init-y-range",
        type=float,
        nargs=3,
        default=[-0.02, 0.42, 5],
        help="[ymin, ymax, num]",
    )

    parser.add_argument(
        "--additional-env-build-kwargs",
        nargs="+",
        action=DictAction,
        help="Additional env build kwargs in xxx=yyy format. If the value "
        'is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument("--logging-dir", type=str, default="./results")
    parser.add_argument("--tf-memory-limit", type=int, default=3072, help="Tensorflow memory limit")
    parser.add_argument("--octo-init-rng", type=int, default=0, help="Octo init rng seed")

    parser.add_argument("--config_path", type=str, default=None, help="path to the config file")
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        nargs="+",
        default="",
        help="checkpoint directory of the training",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="checkpoint directory of the training",
    )
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--double-step", action="store_true")
    parser.add_argument("--execute_step", type=int, default=1, help="execute step for the model")
    args = parser.parse_args()

    # env args: robot pose
    args.robot_init_xs = parse_range_tuple(args.robot_init_x_range)
    args.robot_init_ys = parse_range_tuple(args.robot_init_y_range)
    args.robot_init_quats = []
    for r in parse_range_tuple(args.robot_init_rot_rpy_range[:3]):
        for p in parse_range_tuple(args.robot_init_rot_rpy_range[3:6]):
            for y in parse_range_tuple(args.robot_init_rot_rpy_range[6:]):
                args.robot_init_quats.append((Pose(q=euler2quat(r, p, y)) * Pose(q=args.robot_init_rot_quat_center)).q)
    # env args: object position
    if args.obj_variation_mode == "xy":
        args.obj_init_xs = parse_range_tuple(args.obj_init_x_range)
        args.obj_init_ys = parse_range_tuple(args.obj_init_y_range)
    # update logging info (args.additional_env_save_tags) if using a different camera from default
    if args.obs_camera_name is not None:
        if args.additional_env_save_tags is None:
            args.additional_env_save_tags = f"obs_camera_{args.obs_camera_name}"
        else:
            args.additional_env_save_tags = (args.additional_env_save_tags + f"_obs_camera_{args.obs_camera_name}")

    return args


if __name__ == "__main__":
    CACHE_ROOT = "eval/logs"
    os.system(f"mkdir -p {CACHE_ROOT}")
    os.system(f"chmod 777 {CACHE_ROOT}")

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
    print(configs.keys())
    if "pi0_cfg" in configs.keys():
        print("Changing pi0_cfg path")
        configs["pi0_cfg"] = configs["pi0_cfg"].replace(old_dir + "/vlm4vla", new_dir + "/VLM4VLA")

    args.model_name = configs["config"].split("/")[-1].split(".")[0]
    args.model_name += f'_{configs["exp_name"]}'
    if args.double_step:
        args.model_name += "double"
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
    eval_log_dir = f"{ckpt_dir}/execute_step_{args.execute_step}/{ckpt_path.split('/')[-1].split('.')[0].split('=')[-1]}-eval"
    args.logging_dir = eval_log_dir
    # os.system(f"mkdir {eval_log_dir}")
    os.makedirs(eval_log_dir, exist_ok=True)
    os.system(f"chmod 777 -R {eval_log_dir}")

    model = BaseModelInference(
        ckpt_path=ckpt_path,
        configs=configs,
        device=torch.device("cuda"),
        save_dir=eval_log_dir,
        policy_setup=args.policy_setup,
        execute_step=args.execute_step,
    )

    # run real-to-sim evaluation
    success_arr = maniskill2_evaluator(model, args)
    print(args)
    print(" " * 10, "Average success", np.mean(success_arr))
    json.dump(
        {
            "success_rate": np.mean(success_arr) * 100,
            "test_num": len(success_arr),
            "success_num": int(np.sum(success_arr)),
        }, open(os.path.join(eval_log_dir, f"{args.env_name}_{np.mean(success_arr):.4f}.json"), "w"))
