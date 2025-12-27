import os

os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
os.environ['HF_HOME'] = '/mnt/zjk/jianke_z/huggingface'

import argparse
import json
from pathlib import Path
import importlib
import copy
import functools
from re import L
from typing import Dict, Any, Set, Type
import datetime

from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger, WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.strategies import FSDPStrategy
from lightning import seed_everything
import torch.distributed as dist
import torch.nn as nn

from vlm4vla.train.base_trainer import BaseTrainer
from vlm4vla.data.datamodule.gr_datamodule import GRDataModule
from vlm4vla.data.data_utils import preprocess_image
from vlm4vla.utils.setup_callback import SetupCallback


def get_date_str():
    return str(datetime.date.today())


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def init_lr_monitor_callback():
    return LearningRateMonitor(logging_interval="step")


def init_setup_callback(config):
    return SetupCallback(
        now=str(datetime.datetime.now()).replace(" ", "_"),
        logdir=config["log_dir"],
        ckptdir=config["output_dir"],
        cfgdir=config["log_dir"],
        config=config,
    )


def init_trainer_config(configs):
    # TODO: currently for other strategy we directly use the default settings.
    trainer_config = copy.deepcopy(configs["trainer"])
    trainer_config["devices"] = configs.get("gpus", "auto")
    trainer_config["num_nodes"] = configs.get("num_nodes", 1)
    trainer_config["gradient_clip_val"] = configs.get("gradient_clip_val", 0.0)
    

    # exp_name = configs.get("exp_name", "default")
    # print(exp_name)
    # exp_name = eval(exp_name)

    strategy_name = "None"
    if "strategy" not in trainer_config or trainer_config["strategy"] == "ddp":
        strategy_name = "ddp"
        trainer_config["strategy"] = DDPStrategy(find_unused_parameters=True)
    else:
       strategy_name =  trainer_config.get("strategy", "none")

    if "Qwen" not in configs['model_url']:
        assert configs['model_url'].split('/')[-1] in configs['vlm']['pretrained_model_name_or_path']
        exp_name = f"{configs['model_url'].split('/')[-1]}-bs{configs['batch_size']*configs['num_nodes']*configs['gpus']*configs['trainer']['accumulate_grad_batches']}-lr{configs['learning_rate']}-ws{configs['window_size']}-{configs['act_head']['type']}-latent{configs['act_head']['latent']}"
    else:
        exp_name = f"{'_'.join(configs['vlm']['pretrained_model_name_or_path'].split('/')[4:])}-bs{configs['batch_size']*configs['num_nodes']*configs['gpus']*configs['trainer']['accumulate_grad_batches']}-lr{configs['learning_rate']}-ws{configs['window_size']}-{configs['act_head']['type']}-latent{configs['act_head']['latent']}"
    if "Pi0" in configs["robovlm_name"]:
        exp_name = configs["robovlm_name"] + "_" + exp_name
    if not configs["train_setup"]["train_vision"]:
        exp_name += "-freeze_vision"
    if not configs["train_setup"]["train_text_embedding"]:
        exp_name += "-freeze_textemb"
    if configs["task_name"] == "bridge_finetune":
        exp_name = "bridge_" + exp_name
    elif configs["task_name"] == "fractal_finetune":
        exp_name = "fractal_" + exp_name
    elif configs["task_name"] == "calvin_finetune":
        exp_name = "calvin_" + exp_name
    elif "libero" in configs["task_name"]:
        exp_name = configs["task_name"].split("_")[0] + "_" + exp_name
    elif configs["task_name"] == "realdualarm_finetune":
        exp_name = "realdualarm_" + exp_name
    elif configs["task_name"] == "b1k_finetune":
        exp_name = "b1k" + exp_name
    else:
        raise NotImplementedError
    
    # Add prompt field to exp_name if present
    if configs.get("prompt") is not None:
        exp_name += f"_{configs['prompt']}"
    # init loggers
    loggers = None
    log_dir = Path(os.path.join(get_date_str(), exp_name))
    configs["log_dir"] = configs["log_root"] / log_dir
    if isinstance(trainer_config.get("logger"), list):
        loggers = []
        for logger in trainer_config.get("logger"):
            if logger == "tensorboard":
                loggers.append(TensorBoardLogger(configs["log_dir"].as_posix(), name=exp_name))
            elif logger == "csv":
                loggers.append(CSVLogger(configs["log_dir"].as_posix(), name=exp_name))
            elif logger == "wandb":
                loggers.append(WandbLogger(project="vlm4vla", name=exp_name, save_dir=configs["log_dir"].as_posix()))
            else:
                raise NotImplementedError

    trainer_config["logger"] = loggers

    ckpt_dir = Path(os.path.join(get_date_str(), exp_name))
    configs["output_dir"] = configs["output_root"] / ckpt_dir

    configs["log_dir"].mkdir(parents=True, exist_ok=True)
    configs["output_dir"].mkdir(parents=True, exist_ok=True)
    configs["cache_root"].mkdir(parents=True, exist_ok=True)
    # os.system(f"sudo chmod 777 -R runs/")

    configs["log_dir"] = configs["log_dir"].as_posix()
    configs["output_dir"] = configs["output_dir"].as_posix()
    configs.pop("output_root")
    configs.pop("log_root")
    configs["cache_root"] = configs["cache_root"].as_posix()

    trainer_config["callbacks"] = [
        init_setup_callback(configs),
        init_lr_monitor_callback(),
        ModelCheckpoint(dirpath=configs["output_dir"], save_top_k=-1, every_n_epochs=1),
    ]
    print("trainer_config: ", trainer_config)
    return trainer_config


def experiment(variant):
    seed_everything(variant["seed"] + int(os.environ["RANK"]))
    # import pdb; pdb.set_trace()
    trainer_config = init_trainer_config(variant)
    model_load_path = variant.get("model_load_path", None)

    if trainer_config["strategy"] != "fsdp":
        trainer = Trainer(**trainer_config)
        variant["gpus"] = trainer.num_devices
    
    variant["train_setup"]["precision"] = variant["trainer"]["precision"]

    if variant["fwd_head"] is not None:
        variant["train_setup"]["predict_forward_hand"] = variant["fwd_head"].get("pred_hand_image",
                                                                                 False)  # false by default

    if not os.path.exists(variant['model_path']):
        repo_name = variant["model_url"].split("/")[-1].split(".")[0]
        print(f"VLM backbone does not exist, cloning {variant['model']} from {variant['model_url']}...")
        os.system(f"git clone {variant['model_url']} .vlms/{repo_name}")
        variant['model_path'] = ".vlms/" + repo_name
        variant['model_config'] = os.path.join(variant['model_path'], "config.json")

    if variant["model"] == "kosmos":
        import transformers

        package_dir = transformers.__path__[0]
        os.system("cp tools/modeling_kosmos2.py {}/models/kosmos2/modeling_kosmos2.py".format(package_dir))

        import importlib

        importlib.reload(transformers)

    model = BaseTrainer.from_checkpoint(model_load_path, variant.get("model_load_source", "torch"), variant)
    
    if trainer_config["strategy"] == "fsdp":
        fsdp_wrap_policy = get_wrap_policy_from_model(model.model.backbone)
        trainer_config["strategy"] = FSDPStrategy(
            sharding_strategy="FULL_SHARD",  # 对应 fsdp_sharding_strategy: FULL_SHARD
            # cpu_offload=True,                # 对应 fsdp_offload_params: true
            auto_wrap_policy=fsdp_wrap_policy,  # 自动包装策略
            limit_all_gathers=True,          # 推荐开启以提升通信效率（可选）
            use_orig_params=True,            # 对应 fsdp_use_orig_params: true
            forward_prefetch=True,          # 对应 fsdp_forward_prefetch: false
            backward_prefetch=True,          # 对应 fsdp_backward_prefetch: BACKWARD_PRE（True ≈ BACKWARD_PRE）
            sync_module_states=True,         # 对应 fsdp_sync_module_states: true
        )
        trainer = Trainer(**trainer_config)

    if trainer.precision == "bf16" or trainer.precision == "bf16-mixed":
        import torch
        model = model.to(torch.bfloat16)
    elif trainer.precision == 32:
        import torch
        model = model.to(torch.float32)
    elif trainer.precision == 16 or trainer.precision == "16-mixed":
        import torch
        model = model.to(torch.float16)
    
    if trainer_config["strategy"] != "fsdp":
        # Print trainable and frozen parameters
        # Note: model is BaseTrainer (LightningModule), need to access model.model for actual model parameters
        trainable_params = []
        frozen_params = []
        trainable_param_count = 0
        frozen_param_count = 0
        
        for name, param in model.model.named_parameters():
            if param.requires_grad:
                trainable_params.append(name)
                trainable_param_count += param.numel()
            else:
                frozen_params.append(name)
                frozen_param_count += param.numel()
        
        # Only print on rank 0 to avoid duplicate output in distributed training
        if dist.get_rank() == 0:
            print("=" * 80)
            print(f"训练参数数量: {len(trainable_params)} (总计: {trainable_param_count / 1000000:.2f}M)")
            print(f"冻结参数数量: {len(frozen_params)} (总计: {frozen_param_count / 1000000:.2f}M)")
            print("=" * 80)
            print("\n【训练参数列表】:")
            for i, name in enumerate(trainable_params, 1):
                print(f"  {i}. {name}")
            print("\n" + "=" * 80)
            print("\n【冻结参数列表】:")
            for i, name in enumerate(frozen_params, 1):
                print(f"  {i}. {name}")
            print("=" * 80 + "\n")

    image_preprocess = model.model.image_processor

    _kwargs = {
        "model":
            model,
        "datamodule":
            GRDataModule(
                variant["train_dataset"],
                variant["val_dataset"],
                variant["batch_size"],
                variant["num_workers"],
                tokenizer=model.model.tokenizer,
                tokenizer_config=variant["tokenizer"],
                fwd_pred_next_n=variant["fwd_pred_next_n"],
                window_size=variant["window_size"],
                image_size=variant["image_size"],
                image_fn=functools.partial(
                    preprocess_image,
                    image_processor=image_preprocess,
                    model_type=variant["model"],
                ) if "qwen" not in variant["model"] or "calvin" in variant["task_name"] else (functools.partial(
                    preprocess_image,
                    image_processor=image_preprocess,
                    model_type=variant["model"],
                ), model.model),
                discrete=False,
                discrete_action=False,
                use_mu_law=False,
                mu_val=255,
                n_bin=(256 if variant["act_head"] is None else variant["act_head"].get("n_bin", 256)),
                min_action=(-1 if variant["act_head"] is None else variant["act_head"].get("min_action", -1)),
                max_action=(1 if variant["act_head"] is None else variant["act_head"].get("max_action", 1)),
                discrete_action_history=False,
                act_step=variant.get("fwd_pred_next_n", 1),
                norm_action=variant.get("norm_action", False),
                norm_min=variant.get("norm_min", -1),
                norm_max=variant.get("norm_max", 1),
                regular_action=variant.get("regular_action", False),
                x_mean=variant.get("x_mean", 0),
                x_std=variant.get("x_std", 1),
                weights=variant.get("train_weights", None),
                tcp_rel=False,
                # vit_name=vit_name,
                model_name=variant.get("model", "flamingo"),
            ),
        "ckpt_path":
            variant["resume"],
    }
    if _kwargs["ckpt_path"] is not None:
        print(f"Resuming from {variant['resume']}...")

    trainer.fit(**_kwargs)


def deep_update(d1, d2):
    # use d2 to update d1
    for k, v in d2.items():
        if isinstance(v, dict) and k in d1:
            assert isinstance(d1[k], dict)
            deep_update(d1[k], d2[k])
        else:
            d1[k] = d2[k]
    return d1


def load_config(config_file):
    _config = json.load(open(config_file))
    config = {}
    if _config.get("parent", None):
        deep_update(config, load_config(_config["parent"]))
    deep_update(config, _config)
    return config


def update_configs(configs, args):
    configs["raw_config_path"] = args["config"]
    configs["num_nodes"] = args.get("num_nodes", 1)
    configs["output_root"] = (Path(configs["output_root"]) / configs["model"] / configs["task_name"])
    configs["log_root"] = (Path(configs["log_root"]) / configs["model"] / configs["task_name"])
    configs["cache_root"] = Path(configs["cache_root"]) / configs["model"]

    for k, v in args.items():
        if k not in configs:
            print(f"{k} not in config. The value is {v}.")
            configs[k] = v
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                # assert sub_k in configs[k], f"{sub_k} not in configs {k}"
                if sub_v != None:
                    configs[k][sub_k] = sub_v
        else:
            if v != None:
                configs[k] = v
    return configs


def parse_args():
    parser = argparse.ArgumentParser()

    # Experiment
    parser.add_argument("config", type=str, help="config file used for training")
    parser.add_argument("--gpus", default=1, type=int)
    parser.add_argument("--num_nodes", default=1, type=int)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--log_dir", default=None, type=str)
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--data_dir", default=None, type=str)
    parser.add_argument("--annotation_file", default=None, type=str)
    parser.add_argument("--model_load_path", default=None, type=str)
    parser.add_argument("--data_subfolder", default=None, type=str)
    parser.add_argument("--task_num", default=None, type=int)
    parser.add_argument("--seq_len", default=None, type=float)
    parser.add_argument("--exp_name", default=None, type=str)

    # Loss
    parser.add_argument("--arm_gripper_loss_ratio", default=None, type=float)
    parser.add_argument("--fwd_loss_ratio", default=None, type=float)
    parser.add_argument("--fwd_pred_next_n", default=None, type=int)

    parser.add_argument("--use_multi_modal_emb", default=False, action="store_true")
    parser.add_argument("--no_video_pretrained_model", default=False, action="store_true")
    parser.add_argument("--finetune", default=False, action="store_true")

    # Training
    parser.add_argument("--learning_rate", default=None, type=float)
    parser.add_argument("--min_lr_scale", default=None, type=float)
    parser.add_argument("--warmup_epochs", default=None, type=int)
    parser.add_argument("--weight_decay", default=None, type=float)
    parser.add_argument("--batch_size", default=None, type=int)

    global_names = set(vars(parser.parse_known_args()[0]).keys())

    # Trainer
    trainer_parser = parser.add_argument_group("trainer")
    trainer_parser.add_argument("--strategy", default=None, type=str)
    trainer_parser.add_argument("--precision", default=None, type=str)
    trainer_parser.add_argument("--gradient_clip_val", default=None, type=float)
    trainer_parser.add_argument("--max_epochs", default=None, type=int)
    trainer_names = set(vars(parser.parse_known_args()[0]).keys()) - global_names

    # Model Architecture
    llm_parser = parser.add_argument_group("llm")
    llm_parser.add_argument("--type", default=None, type=str)
    llm_parser.add_argument("--n_embd", default=None, type=int)
    llm_parser.add_argument("--n_layer", default=None, type=int)
    llm_parser.add_argument("--n_head", default=None, type=int)
    llm_names = (set(vars(parser.parse_known_args()[0]).keys()) - global_names - trainer_names)

    args = {}
    trainer_args = {}
    llm_args = {}
    temp_args = vars(parser.parse_args())
    for k, v in temp_args.items():
        if k in global_names:
            args[k] = v
        elif k in trainer_names:
            trainer_args[k] = v
        elif k in llm_names:
            llm_args[k] = v

    args["llm"] = llm_args
    args["trainer"] = trainer_args

    return args


if __name__ == "__main__":
    # import os

    # os.environ['CUDA_LAUNCH_BLOCKING']="1"
    args = parse_args()

    # load config files
    configs = load_config(args.get("config"))
    configs = update_configs(configs, args)

    dist.init_process_group(backend="nccl")
    experiment(variant=configs)

# bash scripts/run.sh configs/calvin_finetune/finetune_paligemma2-3b_calvin.json
# bash scripts/run.sh configs/calvin_finetune/finetune_paligemma2-3b_calvin.json
