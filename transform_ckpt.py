import os
import re
import torch
from robovlms.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict
import shutil


def sort_ckpt(ckpt_dir):
    if isinstance(ckpt_dir, str):
        # get sorted ckpt list
        ckpt_files = os.listdir(ckpt_dir)
        ckpt_files = [f for f in ckpt_files if f.endswith(".ckpt")]
        ckpt_dirs = [ckpt_dir] * len(ckpt_files)

    else:
        # sometimes trials may fail, and the ckpt_dir will be a list of dirs including
        # ckpts from both the original trial and resumed trials
        assert isinstance(ckpt_dir, list)
        ckpt_files = []
        ckpt_dirs = []
        for d in ckpt_dir:
            _ckpt_files = os.listdir(d)
            _ckpt_files = [f for f in _ckpt_files if f.endswith(".ckpt")]
            _ckpt_dirs = [d] * len(_ckpt_files)
            ckpt_files.extend(_ckpt_files)
            ckpt_dirs.extend(_ckpt_dirs)

    ckpt_steps = [re.search(r"step=\d+", f).group() for f in ckpt_files]
    ckpt_steps = [int(s[5:]) for s in ckpt_steps]
    ckpts = list(zip(ckpt_steps, ckpt_dirs, ckpt_files))
    ckpts = sorted(ckpts, key=lambda x: x[0])
    ckpt_files = [os.path.join(x[1], x[2]) for x in ckpts]
    ckpt_steps = [x[0] for x in ckpts]
    return ckpt_files, ckpt_steps


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default="")
    args = parser.parse_args()
    ckpt_dir = args.ckpt_dir
    # ckpt_dir = "/mnt/zjk/jianke_z/RoboVLMs/runs/checkpoints/paligemma/calvin_finetune/2025-07-10/abcd-d-pali2-ws1-FC-latent1-sequencewrong"
    # ckpt_dir = "/mnt/zjk/jianke_z/RoboVLMs/runs/checkpoints/paligemma/calvin_finetune/2025-07-10/abcd-d-pali2-ws1-FC-latent1-sequencewrong/step_0_.ckpt"

    if isinstance(ckpt_dir, str):
        if ckpt_dir.endswith(".ckpt"):
            ckpt_files = [ckpt_dir]
        else:
            ckpt_files, _ = sort_ckpt(ckpt_dir)
    else:
        ckpt_files = ckpt_dir

    for ckpt_path in ckpt_files:
        assert "runs/checkpoints" in ckpt_path
        target_ckpt_path = ckpt_path.replace(".ckpt", ".pt").replace("runs/checkpoints", "runs/torch_checkpoints_fp32")
        os.makedirs(os.path.dirname(target_ckpt_path), exist_ok=True)
        # 复制json文件到目标目录
        try:
            json_path_dir = os.path.dirname(ckpt_path).replace("runs/checkpoints", "runs/logs")

            json_files = [f for f in os.listdir(json_path_dir) if f.endswith(".json")]
            if len(json_files) == 0:
                raise FileNotFoundError(f"在目录 {json_path_dir} 下未找到json文件")
            if len(json_files) > 1:
                print(f"警告：在目录 {json_path_dir} 下找到多个json文件，仅复制第一个: {json_files[0]}")
            target_json_path = os.path.join(os.path.dirname(target_ckpt_path), json_files[0])
            print(f"copying {json_files[0]} to {os.path.dirname(target_json_path)}")
            json_path = os.path.join(json_path_dir, json_files[0])
            if os.path.exists(json_path):
                shutil.copy(json_path, target_json_path)
        except Exception as e:
            print(f"Error copying json file: {e}")

        print(f"converting {ckpt_path} to {target_ckpt_path}")
        convert_zero_checkpoint_to_fp32_state_dict(ckpt_path, target_ckpt_path)
