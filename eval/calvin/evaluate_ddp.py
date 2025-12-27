"""Evaluate the default decision transformer."""

import argparse
import json
import logging
from pathlib import Path
import time
from collections import Counter, defaultdict, namedtuple
from moviepy.editor import ImageSequenceClip
import copy
# from vlm4vla.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict
from tqdm.auto import tqdm
import sys
import os
import hydra
from omegaconf import OmegaConf
from termcolor import colored

sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())
import torch.multiprocessing as mp

# os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ["MESA_GL_VERSION_OVERRIDE"] = "4.1"
import pyrender

from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import (
    collect_plan,
    count_success,
    create_tsne,
    get_default_model_and_env,
    get_env_state_for_initial_condition,
    get_log_dir,
    join_vis_lang,
    print_and_save,
)
from calvin_env.envs.play_table_env import get_env

from pytorch_lightning import seed_everything
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from vlm4vla.utils.config_utils import load_config

from model_wrapper import CustomModel
from eval_utils import print_and_save
from distributed_utils import init_distributed_device, world_info_from_env

logger = logging.getLogger(__name__)

EP_LEN = 360
NUM_SEQUENCES = 1000
# NUM_SEQUENCES = 16

CACHE_ROOT = "eval/logs"

os.system(f"mkdir -p {CACHE_ROOT}")
os.system(f"chmod 777 {CACHE_ROOT}")


def make_env(dataset_path):
    val_folder = Path(dataset_path) / "validation"
    return get_env(val_folder, show_gui=False)


def setup():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def evaluate_policy(
    rank,
    world_size,
    model,
    env,
    eval_sr_path,
    eval_log_dir=None,
    debug=False,
    raw_calvin=False,
    diverse_inst=False,
):
    """Run this function to evaluate a model on the CALVIN challenge."""
    # conf_dir = Path("path/to/calvin/calvin_models") / "conf"
    conf_dir = Path("/home/disk1/jianke_z/VLM4VLA/calvin/calvin_models") / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    if diverse_inst:
        with open("configs/data/calvin/lang_annotation_cache.json", "r") as f:
            val_annotations = json.load(f)
    else:
        val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    eval_log_dir = get_log_dir(eval_log_dir)

    # eval_sequences = get_sequences(NUM_SEQUENCES)
    with open("configs/data/calvin/eval_sequences.json", "r") as f:
        eval_sequences = json.load(f)
    eval_sequences = eval_sequences[:NUM_SEQUENCES]
    eval_sequences = eval_sequences[rank::world_size]
    results = []

    if not debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    sequence_i = rank
    local_i = 0
    for initial_state, eval_sequence in eval_sequences:
        result = evaluate_sequence(
            env,
            model,
            task_oracle,
            initial_state,
            eval_sequence,
            val_annotations,
            debug,
            eval_log_dir,
            sequence_i,
            raw_calvin,
            diverse_inst=diverse_inst,
        )
        results.append(result)
        if not debug:
            success_list = count_success(results)
            with open(eval_sr_path, "a") as f:
                line = f"{local_i}/{NUM_SEQUENCES}: "
                for sr in success_list:
                    line += f"{sr:.3f} | "
                sequence_i += world_size
                local_i += 1
                line += "\n"
                f.write(line)
            eval_sequences.set_description(
                " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(success_list)]) + "|")
        else:
            sequence_i += world_size
            local_i += 1
    return results


def evaluate_policy_ddp(
    rank,
    world_size,
    model,
    env,
    eval_sr_path,
    eval_log_dir=None,
    debug=False,
    raw_calvin=False,
    create_plan_tsne=False,
):
    """
    Run this function to evaluate a model on the CALVIN challenge.

    Args:
        model: Must implement methods of CalvinBaseModel.
        env: (Wrapped) calvin env.
        epoch:
        eval_log_dir: Path where to log evaluation results. If None, logs to /tmp/evaluation/
        debug: If True, show camera view and debug info.
        create_plan_tsne: Collect data for TSNE plots of latent plans (does not work for your custom model)

    Returns:
        Dictionary with results
    """
    epoch = 0
    conf_dir = Path("path/to/calvin/calvin_models") / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    eval_log_dir = get_log_dir(eval_log_dir)

    device_num = int(torch.distributed.get_world_size())
    device_id = torch.distributed.get_rank()
    interval_len = int(NUM_SEQUENCES // device_num)

    eval_sequences = get_sequences(NUM_SEQUENCES)
    eval_sequences = eval_sequences[rank::world_size]
    results = []
    plans = defaultdict(list)
    local_sequence_i = 0
    base_sequence_i = device_id * interval_len

    if not debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    for initial_state, eval_sequence in eval_sequences:
        result = evaluate_sequence(
            env,
            model,
            task_oracle,
            initial_state,
            eval_sequence,
            val_annotations,
            debug,
            eval_log_dir,
            base_sequence_i + local_sequence_i,
            raw_calvin,
        )
        results.append(result)

        success_list = count_success(results)
        with open(eval_sr_path, "a") as f:
            line = f"{local_sequence_i}/{interval_len}: "
            for sr in success_list:
                line += f"{sr:.3f} | "
            line += "\n"
            f.write(line)
        eval_sequences.set_description(" ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(success_list)]) +
                                       "|")

        local_sequence_i += 1

    def merge_multi_list(res):
        tmp = []
        for l in res:
            tmp.extend(l)
        return tmp

    def extract_iter_from_tqdm(tqdm_iter):
        return [_ for _ in tqdm_iter]

    if create_plan_tsne:
        create_tsne(plans, eval_log_dir, epoch)

    return results


def evaluate_sequence(
    env,
    model,
    task_checker,
    initial_state,
    eval_sequence,
    val_annotations,
    debug,
    eval_log_dir,
    sequence_i,
    raw_calvin,
    diverse_inst=False,
):
    """Evaluates a sequence of language instructions."""
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter = 0
    if debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    for subtask_i, subtask in enumerate(eval_sequence):
        success = rollout(
            env,
            model,
            task_checker,
            subtask,
            val_annotations,
            debug,
            eval_log_dir,
            subtask_i,
            sequence_i,
            raw_calvin,
            diverse_inst=diverse_inst,
        )
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter


def rollout(
    env,
    model,
    task_oracle,
    subtask,
    val_annotations,
    debug,
    eval_log_dir,
    subtask_i,
    sequence_i,
    raw_calvin=False,
    diverse_inst=False,
):
    """Run the actual rollout on one subtask (which is one natural language instruction)."""
    if debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)
        img_list = []
    obs = env.get_obs()
    # get lang annotation for subtask
    if diverse_inst:
        lang_annotation = val_annotations[sequence_i][subtask_i]
    else:
        lang_annotation = val_annotations[subtask][0]
    model.reset()
    start_info = env.get_info()

    for _ in range(EP_LEN):
        action = model.step(obs, lang_annotation,execute_step=10)
        obs, _, _, current_info = env.step(action)
        if debug:
            img_copy = copy.deepcopy(obs["rgb_obs"]["rgb_static"])
            img_list.append(img_copy)

        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            if debug:
                print(colored("success", "green"), end=" ")
                clip = ImageSequenceClip(img_list, fps=30)
                clip.write_gif(
                    os.path.join(eval_log_dir, f"{sequence_i}-{subtask_i}-{subtask}-succ.gif"),
                    fps=30,
                )
            return True

    if debug:
        print(colored("fail", "red"), end=" ")
        clip = ImageSequenceClip(img_list, fps=30)
        clip.write_gif(
            os.path.join(eval_log_dir, f"{sequence_i}-{subtask_i}-{subtask}-fail.gif"),
            fps=30,
        )
    return False


def parser_args():
    seed_everything(0, workers=True)  # type:ignore
    parser = argparse.ArgumentParser(description="Evaluate a trained model on multistep sequences with language goals.")
    parser.add_argument(
        "--dataset_path",
        default="/home/disk1/jianke_z/task_ABC_D",
        type=str,
        help="Path to the dataset root directory.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug info and visualize environment.",
    )

    # yaml_path takes the highest priority, then the log_dir, finally the config_path
    parser.add_argument("--config_path", type=str, default=None, help="path to the config file")
    parser.add_argument(
        "--is_pt_config",
        action="store_true",
        help="whether the specified config path is a pretrain config file.",
    )

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
    parser.add_argument("--ckpt_idx", type=int, default=-1, help="which ckpt is going to be evaluated")
    parser.add_argument("--device_id", default=0, type=int, help="CUDA device")
    parser.add_argument("--no_cache", action="store_false")
    parser.add_argument("--raw_calvin", action="store_true")
    parser.add_argument("--debug_model", action="store_true")
    parser.add_argument("--diverse_inst", action="store_true")

    args = parser.parse_args()
    return args


def main():
    args = parser_args()
    setup()
    config_path = args.config_path
    ckpt_dir = args.ckpt_dir
    ckpt_idx = args.ckpt_idx

    # Loading configs
    assert config_path != None
    # Load config file
    configs = load_config(config_path)

    # change all path in configs to new path (for testing on another machine)
    # new_dir = "/home/disk1/jianke_z"
    # old_dir = "/mnt/zjk/jianke_z"
    # keys_to_change = [
    #     "model_path", "model_config", ["tokenizer", "pretrained_model_name_or_path"],
    #     ["vlm", "pretrained_model_name_or_path"], ["train_dataset", "data_dir"], ["val_dataset", "data_dir"]
    # ]

    # def check_qwen_dir(path):
    #     if not os.path.exists(path):
    #         if "Qwen2.5" in path or "qwen25" in path or "Qwen-2.5" in path or "qwen-2.5" in path:
    #             has_configjson = "config.json" in path
    #             if "3b" in path or "3B" in path:
    #                 new_path = "/home/disk1/jianke_z/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/c747f21f03e7d0792c30766310bd7d8de17eeeb3"
    #             elif "7b" in path or "7B" in path:
    #                 new_path = "/home/disk1/jianke_z/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5"
    #             else:
    #                 raise ValueError(f"Unknown Qwen2.5 model size in {configs[key[0]][key[1]]}")
    #             if has_configjson:
    #                 new_path += "/config.json"
    #             return new_path
    #     else:
    #         return path

    # for key in keys_to_change:
    #     if isinstance(key, list):
    #         configs[key[0]][key[1]] = configs[key[0]][key[1]].replace(old_dir, new_dir)
    #         if "dataset" in key[0]:
    #             configs[key[0]][key[1]] = configs[key[0]][key[1]].replace("data/robotics_0707/", "")
    #         configs[key[0]][key[1]] = check_qwen_dir(configs[key[0]][key[1]])
    #     else:
    #         configs[key] = configs[key].replace(old_dir, new_dir)
    #         configs[key] = check_qwen_dir(configs[key])

    # if "pi0_cfg" in configs.keys():
    #     print("Changing pi0_cfg path")
    #     configs["pi0_cfg"] = configs["pi0_cfg"].replace(old_dir + "/vlm4vla", new_dir + "/VLM4VLA")

    if args.is_pt_config:
        exp_name = os.path.basename(config_path)
        configs["exp_name"] = exp_name
        # generate ft configs from pt configs
        from vlm4vla.utils.config_utils import generate_calvin_ft_configs, deep_update

        ft_configs = generate_calvin_ft_configs(configs)
        deep_update(configs, ft_configs)

    # Get checkpoint path
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
    # if os.path.isdir(ckpt_path):
    #     target_ckpt_path = ckpt_path.replace(".ckpt", ".pt")
    #     if dist.get_rank() == 0:
    #         print(f"converting {ckpt_path} to {target_ckpt_path}")
    #         convert_zero_checkpoint_to_fp32_state_dict(ckpt_path, target_ckpt_path)

    #     dist.barrier()
    #     ckpt_path = target_ckpt_path

    from vlm4vla.utils.config_utils import get_exp_name

    eval_exp_name = get_exp_name(f"{os.path.basename(config_path)}", mode="eval")
    if args.no_cache:
        eval_log_dir = f"{ckpt_dir}/{ckpt_path.split('/')[-1].split('.')[0].split('=')[-1]}-eval"
    else:
        eval_log_dir = os.path.join(CACHE_ROOT, eval_exp_name)
    os.system(f"mkdir {eval_log_dir}")
    os.system(f"chmod 777 -R {eval_log_dir}")

    args.local_rank, args.rank, args.world_size = world_info_from_env()
    model = CustomModel(
        ckpt_path=ckpt_path,
        configs=configs,
        device=torch.device("cuda"),
        save_dir=eval_log_dir,
        raw_calvin=args.raw_calvin,
        debug=args.debug_model,
    )
    args.dataset_path = "/".join(configs["val_dataset"]["data_dir"].split("/")[:-1])
    env = make_env(args.dataset_path)

    # TODO check if this code is needed
    ckpt_step = ckpt_path.split("/")[-1].split(".")[0]
    if args.diverse_inst:
        suffix = "_enrich"
    else:
        suffix = ""
    sr_path = os.path.join(eval_log_dir, f"success_rate_calvin_{ckpt_step}{suffix}.txt")
    result_path = os.path.join(eval_log_dir, f"results_calvin_{ckpt_step}{suffix}_rand-{args.rank}.json")
    cache_file = os.path.join(eval_log_dir, f"meta_info_step_{ckpt_step}{suffix}.json")

    if not args.no_cache and args.local_rank == 0:
        if os.path.exists(cache_file):
            os.system(f"rm {cache_file}")
        with open(cache_file, "w") as f:
            _info = {
                "eval_sr_path": sr_path,
                "eval_result_path": result_path,
                "eval_log_dir": eval_log_dir,
                "raw_config_path": configs.get("raw_config_path", None),
            }
            json.dump(_info, f, indent=2)

    results = evaluate_policy(
        args.rank,
        args.world_size,
        model,
        env,
        eval_sr_path=sr_path,
        eval_log_dir=ckpt_dir,
        debug=args.debug,
        diverse_inst=args.diverse_inst,
    )

    # gather results
    with open("configs/data/calvin/eval_sequences.json", "r") as f:
        eval_sequences = json.load(f)
    eval_sequences = eval_sequences[:NUM_SEQUENCES]
    eval_sequences = eval_sequences[args.rank::args.world_size]
    print_and_save(results, eval_sequences, result_path, None)

    # if args.no_cache and args.local_rank == 0:
    #     os.system("rm -r ./temp/")

    dist.destroy_process_group()


if __name__ == "__main__":
    os.environ["NCCL_BLOCKING_WAIT"] = "1"
    main()
