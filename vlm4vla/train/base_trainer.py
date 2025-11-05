import time
import warnings
import os
import json
import numpy as np
from functools import partial
import math

import torch
import torch.nn.functional as F
import torch.distributed as dist
import lightning.pytorch as pl

from vlm4vla.utils.dist_train import get_rank
from vlm4vla.train.train_utils import adjust_learning_rate
import vlm4vla.model.backbone as RoboVLM_Backbone


class BaseTrainer(pl.LightningModule):

    def __init__(self, configs):
        super().__init__()
        self._main_rank_print("--------------- model configs ---------------")
        self._main_rank_print(configs)
        self.configs = configs
        self.model_fn = getattr(RoboVLM_Backbone, configs["robovlm_name"])
        self._initialize()
        self.save_hyperparameters()
        if isinstance(configs["val_dataset"], list):
            self.val_set_names = [self._parse_dataset_name(cfg) for cfg in configs["val_dataset"]]
        elif isinstance(configs["val_dataset"], dict):
            # FIXME: hotfix
            self.val_set_names = None
        else:
            raise NotImplementedError

    def _init_policy(self):
        print("fwd next n: {}".format(self.configs["fwd_pred_next_n"]))

        model = self.model_fn(
            configs=self.configs,
            train_setup_configs=self.configs["train_setup"],
            fwd_head_configs=self.configs["fwd_head"],
            window_size=self.configs["window_size"],
            use_hand_rgb=self.use_hand_rgb,
            act_head_configs=self.configs["act_head"],
            fwd_pred_next_n=self.configs["fwd_pred_next_n"],
            use_vision_resampler=self.configs.get("use_vision_resampler", False),
            vision_resampler_configs=self.configs.get("vision_resampler", None),
            use_clip_norm=self.configs.get("use_clip_norm", False),
            # use_state=self.configs.get("use_state, False", False),
            use_state=self.configs.get("use_state", False),
        )
        model = model.train()
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # print({k for k, v in model.named_parameters() if v.requires_grad})
        self._main_rank_print(f"Trainable Model Parameters: {total_params / 1000000:.2f}M")
        if len(list(model.named_children())) == 1:
            for name, param in model.model.named_children():
                total_params = sum(p.numel() for p in param.parameters() if p.requires_grad)
                self._main_rank_print(f"-- trainable {name} Parameters: {total_params / (1000*1000):.2f}M")
        else:
            for name, param in model.named_children():
                total_params = sum(p.numel() for p in param.parameters() if p.requires_grad)
                self._main_rank_print(f"-- trainable {name} Parameters: {total_params / (1000*1000):.2f}M")
        # assert False
        return model

    def _parse_dataset_name(self, dataset_config):
        dataset_path = dataset_config["data_dir"]
        avail_dataset = ["mode1", "mode3", "bridge", "rt-1", "ego4d", "calvin"]
        for name in avail_dataset:
            if name in dataset_path.lower():
                return name
        return "UNKNOWN_DATA"

    @staticmethod
    def _main_rank_print(*args, **kwargs):
        if get_rank() == 0:
            print(*args, **kwargs)

    @property
    def num_gpus(self):
        return self.trainer.num_devices * self.trainer.num_nodes

    def _initialize(self):
        self.use_hand_rgb = self.configs["use_hand_rgb"]
        self.use_multi_modal_emb = self.configs.get("use_multi_modal_emb", False)
        self.finetune = self.configs.get("finetune", False)
        self.no_video_pretrained_model = self.configs.get("no_video_pretrained_model", False)

        self.model = self._init_policy()
        self.cap_loss_ratio = self.configs["cap_loss_ratio"]  # 0.05 by default
        self.arm_gripper_loss_ratio = self.configs["arm_gripper_loss_ratio"]  # 0.01 by default
        self.fwd_loss_ratio = self.configs["fwd_loss_ratio"]  # 0 by default
        self.kl_div_ratio = self.configs.get("kl_div_ratio", 0.05)  # 0.05 by default
        self.clip_norm_ratio = self.configs.get("clip_norm_ratio", 0.05)

        self.vl_cotrain_ratio = self.configs.get("vl_cotrain_ratio", 0.05)
        if (self.configs["act_head"] is not None and self.configs["act_head"]["action_space"] == "discrete"):
            self.vl_cotrain_ratio *= 10

        self.act_pred = self.configs["train_setup"]["predict_action"]
        self.fwd_pred = self.configs["train_setup"]["predict_forward"]
        self.fwd_pred_hand = self.configs["train_setup"]["predict_forward_hand"]
        self.cap_pred = self.configs["train_setup"]["predict_caption"]
        # self.use_hand_rgb = self.model.use_hand_rgb

        # Make sure that at least one prediction flag is on
        # assert self.act_pred or self.fwd_pred or self.cap_pred

        self.start_time = time.time()

    @staticmethod
    def get_converted_fp32_paths(deepspeed_ckpt_path):
        deepspeed_ckpt_path = deepspeed_ckpt_path.rstrip("/")
        ckpt_dir = os.path.dirname(deepspeed_ckpt_path)
        ckpt_name = os.path.basename(deepspeed_ckpt_path)
        fp32_ckpt_name = f"{ckpt_name}.fp32.pt"
        converted_path = os.path.join(ckpt_dir, fp32_ckpt_name)
        return converted_path

    @classmethod
    def from_checkpoint(cls, ckpt_path=None, ckpt_source="torch", configs=None):
        if ckpt_path is None:
            assert (configs is not None), "ckpt_path and configs are both None for initialization."
            return cls(configs)

        if os.path.isdir(ckpt_path):
            ckpt_source = "deepspeed"

        if ckpt_source == "torch":
            assert configs is not None, ("to initialize the model with a torch pretrained state dict, "
                                         "you need to specify the configs for initialization.")
            model = cls(configs)
            checkpoint = torch.load(configs["model_load_path"], map_location="cpu")

            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                raise KeyError(f"The checkpoint must has state_dict or model_state_dict as key")

            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            del checkpoint
            from vlm4vla.train.train_utils import convert_old_state_dict

            state_dict = convert_old_state_dict(state_dict)
            try:
                msg = model.load_state_dict(state_dict, strict=False)
                cls._main_rank_print(msg)
            except:
                del state_dict["act_head"]
                msg = model.load_state_dict(state_dict, strict=False)
                cls._main_rank_print(msg)
            del state_dict
            return model

        if ckpt_source == "lightning":
            checkpoint = torch.load(ckpt_path, map_location="cpu")

            model = cls(configs)
            msg = model.load_state_dict(checkpoint["state_dict"], strict=False)
            cls._main_rank_print(msg)
            return model

        if ckpt_source == "deepspeed":
            # FIXME: currently I don't find a proper way to load sharded DeepSpeed model using pytorch lightning.
            #   Here the solution is to convert the DeepSpeed model to FP32, then just load it as a torch model.

            # convert deepspeed checkpoint to lightning
            coverted_path = cls.get_converted_fp32_paths(ckpt_path)

            if not os.path.exists(coverted_path):
                if dist.get_rank() == 0:
                    from vlm4vla.utils.zero_to_fp32 import (
                        convert_zero_checkpoint_to_fp32_state_dict,)

                    print(f"converting {ckpt_path} to {coverted_path}")
                    convert_zero_checkpoint_to_fp32_state_dict(ckpt_path, coverted_path)
                dist.barrier()

            # assert os.path.exists(coverted_path), \
            #     "Please use tools/convert_deepspeed_to_fp32.py [DEEPSPEED_CKPT]" \
            #     "for checkpoint conversion first."

            # remove unpretrained params
            cls._main_rank_print(f"Loading pretrained model from {coverted_path}...")
            checkpoint = torch.load(coverted_path, map_location="cpu")

            model = cls(configs)
            msg = model.load_state_dict(checkpoint["state_dict"], strict=False)
            cls._main_rank_print(msg)
            return model

        raise NotImplementedError("Unknown source of checkpoints. Legal choices: torch, lightning, or deepspeed.")

    def configure_optimizers(self):
        eff_batch_size = (self.configs["batch_size"] * self.num_gpus * (self.configs["seq_len"] - 1))
        eff_lr = self.configs["learning_rate"]
        self._main_rank_print("-" * 40)
        self._main_rank_print("LR SCHEDULER CONFIGS:")
        self._main_rank_print(f"effective batch size: {eff_batch_size}, effective learning rate: {eff_lr}")

        # optimizer = torch.optim.AdamW(self.get_grouped_params(self.model))
        optimizer = torch.optim.AdamW(self.get_grouped_params(self.model), lr=eff_lr)

        assert self.trainer.max_epochs is not None
        num_training_batches = self.trainer.estimated_stepping_batches
        iter_per_epoch = num_training_batches / self.trainer.max_epochs

        warmup_epochs = self.configs.get("warmup_epochs", 0)
        warmup_steps = self.configs.get("warmup_steps", 0)
        max_iters = 1e10
        if self.configs["trainer"]["max_steps"] != -1:
            max_iters = self.configs["trainer"]["max_steps"]
        lr_scheduler_configs = {
            "warmup_iters": warmup_epochs * iter_per_epoch + warmup_steps,
            "iters": min(self.configs["trainer"]["max_epochs"] * iter_per_epoch, max_iters),
            "min_lr_scale": self.configs["min_lr_scale"],
        }
        self._main_rank_print(lr_scheduler_configs)

        scheduler_type = self.configs.get("scheduler", "constant")

        from transformers.optimization import (
            get_constant_schedule_with_warmup,
            get_cosine_schedule_with_warmup,
        )

        if scheduler_type == "constant":
            scheduler = get_constant_schedule_with_warmup(optimizer, lr_scheduler_configs["warmup_iters"])
        elif scheduler_type == "half-cosine":
            lr_lambda = partial(adjust_learning_rate, configs=lr_scheduler_configs)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        elif scheduler_type == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                lr_scheduler_configs["warmup_iters"],
                num_training_steps=lr_scheduler_configs["iters"],
            )
        else:
            raise NotImplementedError

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def _get_loss(self, prediction):
        # print(prediction)
        loss_arm_act = prediction.get("loss_arm_act", None)
        loss_gripper_act = prediction.get("loss_gripper_act", None)
        loss_obs = prediction.get("loss_obs_fwd", None)
        loss_hand_obs = prediction.get("loss_hand_obs_fwd", None)
        acc_arm_act = prediction.get("acc_arm_act", None)
        acc_gripper_act = prediction.get("acc_gripper_act", None)
        loss_cap = prediction.get("loss_cap", None)
        loss_kl = prediction.get("loss_kl", None)
        loss_vl_cotrain = prediction.get("loss_vl_cotrain", None)

        ### loss logout for discrete setting
        action_l1 = prediction.get("action_l1_act", None)

        clip_l1 = prediction.get("text_l1_clip", None)

        loss = torch.tensor(0.0).to(self.device)
        if self.act_pred:
            loss_act = (loss_arm_act if loss_arm_act is not None else 0) + (
                loss_gripper_act * self.arm_gripper_loss_ratio if loss_gripper_act is not None else 0)
            loss += loss_act
            if loss_kl is not None:
                loss += self.kl_div_ratio * loss_kl
            if clip_l1 is not None:
                loss += self.clip_norm_ratio * clip_l1
        else:
            loss_act = None

        if self.fwd_pred:
            loss += self.fwd_loss_ratio * (loss_obs if loss_obs is not None else 0)
            if self.fwd_pred_hand:
                loss += self.fwd_loss_ratio * (loss_hand_obs if loss_hand_obs is not None else 0)
        if loss_cap is not None:
            loss += self.cap_loss_ratio * loss_cap

        if loss_vl_cotrain is not None:
            loss += self.vl_cotrain_ratio * loss_vl_cotrain

        output = {
            "loss": loss,
            "loss_act": loss_act,
            "loss_arm_act": loss_arm_act,
            "loss_gripper_act": loss_gripper_act,
            "acc_arm_act": acc_arm_act,
            "acc_gripper_act": acc_gripper_act,
            "loss_obs": loss_obs,
            "loss_hand_obs": loss_hand_obs,
            "action_l1": action_l1,
            "loss_kl": loss_kl,
            "clip_l1": clip_l1,
            "loss_vl_cotrain": loss_vl_cotrain,
        }

        return output

    def _log_output(self, output, phase, prog_bar_set=None, dataset=None, **kwargs):
        prog_bar_set = prog_bar_set or {}
        for k, v in output.items():
            if v is None:
                continue

            log_name = f"{phase}_{k}"
            if dataset is not None:
                log_name = f"{dataset}_{log_name}"

            if k in prog_bar_set:
                self.log(log_name, v, prog_bar=True, **kwargs)
            else:
                self.log(log_name, v, **kwargs)

    def _process_batch(self, batch):
        """
        Action Prediction:
            args: rgb, language, attention_mask, hand_rgb, action
            reformat: action to input and target (seq_len = window size + chunck size)
        Video Prediction:
            args: rgb, language, attention mask, hand_rgb
            reformat: rgb, [hand_rgb] to input and target (seq_len = window size + chunck size)
        Video Caption:
            args: rgb, language, attention_mask
            reformat: Identity
        Image Caption:
            args: rgb, language, attention_mask
            reformat: Identity
            seq_len = 1
        """

        # print(type(batch), len(batch))
        if isinstance(batch, list):
            # print(batch[0].keys())
            batch = batch[0]
        if isinstance(batch["rgb"], list):
            rgb = [_.cuda() for _ in batch["rgb"]]
        else:
            rgb = batch["rgb"].cuda()
            if len(rgb.shape) == 4:
                rgb = rgb.unsqueeze(1)
            assert len(rgb.shape) == 5
        # rgb: default：16,1,3,224,224 for paligemma, 0-1预处理好; 16,1,3,224,224 for qwen25vl, 0-255未进行预处理

        if isinstance(batch["text"], list) and isinstance(batch["text"][0], str):
            seq_len = self.configs["window_size"]
            assert not self.use_hand_rgb
            assert batch["text_mask"] is None

            from torchvision.transforms.functional import to_pil_image
            image_inputs = []  # 16*seq_len, PIL.Image
            text = []
            for i in range(rgb.shape[0]):
                for j in range(seq_len):
                    image_inputs.append(to_pil_image(rgb[i][j].cpu().to(torch.uint8)))
                    text.append(batch["text"][i])
            image_inputs = self.model.process_vision_info(image_inputs)  # 16,1,3,224,224
            # to_pil_image(rgb[0][0].cpu().to(torch.uint8)).save("test.png")
            self.model.tokenizer.padding_side = "right"
            inputs = self.model.processor(
                text=text,
                images=image_inputs,
                videos=None,
                padding=True,
                return_tensors="pt",
            )
            for k, v in inputs.items():
                # qwen has input_ids, attention_mask, pixel_values, image_grid_thw
                inputs[k] = v.cuda()
            text_mask = inputs.attention_mask
            language = inputs
        elif isinstance(batch["text"], torch.Tensor):
            seq_len = self.configs["window_size"]
            language = batch["text"].cuda()
            text_mask = batch["text_mask"].cuda()
        elif "attention_mask" in batch["text"].keys():
            # 加速qwenvl的预处理, text里是处理好的inputs
            seq_len = self.configs["window_size"]
            inputs = batch["text"]
            for k, v in inputs.items():
                # qwen has input_ids, attention_mask, pixel_values, image_grid_thw
                inputs[k] = v.cuda()
            text_mask = inputs.attention_mask
            language = inputs

        if batch.get("action", None) is not None:
            action = batch["action"].cuda()
        else:
            action = None

        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = batch["attention_mask"].cuda()

        if self.use_hand_rgb and batch.get("hand_rgb", None) is not None:
            hand_rgb = batch["hand_rgb"].cuda()
        else:
            hand_rgb = None

        # Split arm and gripper action
        arm_action = None
        gripper_action = None

        if action is not None:
            arm_action = action[:, :, :6]  # b,len,act_dim-1
            gripper_action = action[:, :, 6]  # b,len
            gripper_action = (gripper_action + 1.0) / 2
            gripper_action = gripper_action.long()

        fwd_rgb_chunck = batch.get("fwd_rgb_chunck", None)
        fwd_hand_rgb_chunck = batch.get("fwd_hand_rgb_chunck", None)
        if fwd_rgb_chunck is not None:
            fwd_rgb_chunck = fwd_rgb_chunck.cuda()
        if fwd_hand_rgb_chunck is not None:
            fwd_hand_rgb_chunck = fwd_hand_rgb_chunck.cuda()

        arm_action_chunck = None
        gripper_action_chunck = None
        action_chunck = batch.get("action_chunck", None)
        if action_chunck is not None:
            action_chunck = action_chunck.cuda()
            arm_action_chunck = action_chunck[..., :6]
            gripper_action_chunck = action_chunck[..., -1]

        if isinstance(rgb, torch.Tensor):
            rgb = rgb[:, :seq_len]
            if hand_rgb is not None:
                hand_rgb = hand_rgb[:, :seq_len]

        chunck_mask = batch.get("chunck_mask", None)
        if chunck_mask is not None:
            chunck_mask = chunck_mask.cuda()

        fwd_mask = batch.get("fwd_mask", None)
        if fwd_mask is not None:
            fwd_mask = fwd_mask.bool().cuda()

        # data preparation for discrete action inputs and labels
        instr_and_action_ids = batch.get("instr_and_action_ids", None)
        if instr_and_action_ids is not None:
            instr_and_action_ids = instr_and_action_ids.cuda()

        instr_and_action_labels = batch.get("instr_and_action_labels", None)
        if instr_and_action_labels is not None:
            instr_and_action_labels = instr_and_action_labels.cuda()

        instr_and_action_mask = batch.get("instr_and_action_mask", None)
        if instr_and_action_mask is not None:
            instr_and_action_mask = instr_and_action_mask.cuda()

        rel_state = batch.get("rel_state", None)
        # clip_text_ids = batch.get('clip_text', None)
        raw_text = batch.get("raw_text", None)
        data_source = batch.get("data_source", "calvin_action")
        # import pdb; pdb.set_trace()
        return (
            rgb,
            hand_rgb,
            attention_mask,
            language,
            text_mask,
            fwd_rgb_chunck,
            fwd_hand_rgb_chunck,
            arm_action,
            gripper_action,
            arm_action_chunck,
            gripper_action_chunck,
            chunck_mask,
            fwd_mask,
            instr_and_action_ids,
            instr_and_action_labels,
            instr_and_action_mask,
            raw_text,
            rel_state,
            data_source,
        )

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        with torch.no_grad():
            if isinstance(batch, tuple):
                batch = batch[0]
            (
                rgb,
                hand_rgb,
                attention_mask,
                language,
                text_mask,
                fwd_rgb_chunck,
                fwd_hand_rgb_chunck,
                arm_action,
                gripper_action,
                arm_action_chunck,
                gripper_action_chunck,
                chunck_mask,
                fwd_mask,
                instr_and_action_ids,
                instr_and_action_labels,
                instr_and_action_mask,
                raw_text,
                rel_state,
                data_source,
            ) = self._process_batch(batch)
            # import pdb; pdb.set_trace()
            prediction = self.model.forward(
                rgb,
                language,
                attention_mask=text_mask,
                action_labels=(arm_action_chunck, gripper_action_chunck),
                action_mask=chunck_mask,
                caption_labels=(language.clone() if (isinstance(rgb, torch.Tensor) and rgb.shape[1] == 1 and
                                                     isinstance(language, torch.Tensor)) else None),
                caption_mask=(text_mask.clone() if (isinstance(rgb, torch.Tensor) and rgb.shape[1] == 1 and
                                                    isinstance(text_mask, torch.Tensor)) else None),
                vision_gripper=hand_rgb,
                fwd_rgb_labels=fwd_rgb_chunck,
                fwd_hand_rgb_labels=fwd_hand_rgb_chunck,
                fwd_mask=fwd_mask,
                instr_and_action_ids=instr_and_action_ids,
                instr_and_action_labels=instr_and_action_labels,
                instr_and_action_mask=instr_and_action_mask,
                raw_text=raw_text,
                data_source=data_source,
                rel_state=rel_state,
                mode="val",
            )

            output = self._get_loss(prediction)
            # process other log like eval_acc_threshold,eval_l1

            prog_bar_set = {"loss"}
            if self.act_pred:
                prog_bar_set.add("loss_arm_act")  # used
                prog_bar_set.add("loss_gripper_act")  # used
                prog_bar_set.add("acc_arm_act")
                prog_bar_set.add("acc_gripper_act")  # used
                prog_bar_set.add("action_l1")
                prog_bar_set.add("clip_l1")  # used
            if self.fwd_pred:
                prog_bar_set.add("loss_obs_fwd")
                if self.fwd_pred_hand:
                    prog_bar_set.add("loss_hand_obs_fwd")
            keys = prediction.keys()
            for key in keys:
                if key not in output:
                    output[key] = prediction[key]
                    prog_bar_set.add(key)

            dataset = None
            if self.val_set_names is not None:
                dataset = self.val_set_names[dataloader_idx]

            self._log_output(
                output,
                phase="val",
                prog_bar_set=prog_bar_set,
                sync_dist=True,
                on_epoch=True,
                on_step=False,
                dataset=dataset,
            )

    def training_step(self, batch, batch_idx):
        # print(type(batch))
        # print('-'*100)
        # import pdb; pdb.set_trace()
        if isinstance(batch, tuple):
            batch = batch[0]
        (
            rgb,
            hand_rgb,
            attention_mask,
            language,
            text_mask,
            fwd_rgb_chunck,
            fwd_hand_rgb_chunck,
            arm_action,
            gripper_action,
            arm_action_chunck,
            gripper_action_chunck,
            chunck_mask,
            fwd_mask,
            instr_and_action_ids,
            instr_and_action_labels,
            instr_and_action_mask,
            raw_text,
            rel_state,
            data_source,
        ) = self._process_batch(batch)
        # if self.global_rank == 0:
        #     param_names = [
        #         "rgb", "hand_rgb", "attention_mask", "language", "text_mask", "fwd_rgb_chunck", "fwd_hand_rgb_chunck",
        #         "arm_action", "gripper_action", "arm_action_chunck", "gripper_action_chunck", "chunck_mask", "fwd_mask",
        #         "instr_and_action_ids", "instr_and_action_labels", "instr_and_action_mask", "raw_text", "rel_state",
        #         "data_source"
        #     ]
        #     params = [
        #         rgb, hand_rgb, attention_mask, language, text_mask, fwd_rgb_chunck, fwd_hand_rgb_chunck, arm_action,
        #         gripper_action, arm_action_chunck, gripper_action_chunck, chunck_mask, fwd_mask, instr_and_action_ids,
        #         instr_and_action_labels, instr_and_action_mask, raw_text, rel_state, data_source
        #     ]
        #     print("输入参数类型信息：")
        #     for name, param in zip(param_names, params):
        #         if isinstance(param, torch.Tensor):
        #             try:
        #                 print(f"{name}: torch.Tensor, shape: {tuple(param.shape)}, value: {param.mean()}")
        #             except:
        #                 print(f"{name}: torch.Tensor, shape: {tuple(param.shape)}, value: {param[0]}")
        #         else:
        #             print(f"{name}: {type(param)}, value: {param}")
        prediction = self.model.forward(
            rgb,
            language,
            attention_mask=text_mask,
            action_labels=(arm_action_chunck, gripper_action_chunck),
            action_mask=chunck_mask,
            caption_labels=(language.clone() if (isinstance(rgb, torch.Tensor) and rgb.shape[1] == 1 and
                                                 isinstance(language, torch.Tensor)) else None),
            caption_mask=(text_mask.clone() if (isinstance(rgb, torch.Tensor) and rgb.shape[1] == 1 and
                                                isinstance(text_mask, torch.Tensor)) else None),
            vision_gripper=hand_rgb,
            fwd_rgb_labels=fwd_rgb_chunck,
            fwd_hand_rgb_labels=fwd_hand_rgb_chunck,
            fwd_mask=fwd_mask,
            instr_and_action_ids=instr_and_action_ids,
            instr_and_action_labels=instr_and_action_labels,
            instr_and_action_mask=instr_and_action_mask,
            raw_text=raw_text,
            data_source=data_source,
            rel_state=rel_state,
            mode="train",
        )

        output = self._get_loss(prediction)

        prog_bar_set = {"loss"}
        if self.act_pred:
            prog_bar_set.add("loss_arm_act")
            prog_bar_set.add("loss_gripper_act")
            prog_bar_set.add("acc_arm_act")
            prog_bar_set.add("acc_gripper_act")
            prog_bar_set.add("action_l1")
            prog_bar_set.add("clip_l1")
        if self.fwd_pred:
            prog_bar_set.add("loss_obs")
            if self.fwd_pred_hand:
                prog_bar_set.add("loss_hand_obs")
        prog_bar_set.add("loss_vl_cotrain")

        self._log_output(
            output,
            phase="train",
            prog_bar_set=prog_bar_set,
            on_step=True,
            on_epoch=False,
        )
        return output["loss"]

    def inference_step(self, batch):
        with torch.no_grad():
            # import pdb; pdb.set_trace()
            (
                rgb,
                hand_rgb,
                attention_mask,
                language,
                text_mask,
                fwd_rgb_chunck,
                fwd_hand_rgb_chunck,
                arm_action,
                gripper_action,
                arm_action_chunck,
                gripper_action_chunck,
                chunck_mask,
                fwd_mask,
                instr_and_action_ids,
                instr_and_action_labels,
                instr_and_action_mask,
                raw_text,
                rel_state,
                data_source,
            ) = self._process_batch(batch)

            prediction = self.model.inference(
                rgb,
                language,
                attention_mask=text_mask,
                action_labels=(arm_action_chunck, gripper_action_chunck),
                action_mask=chunck_mask,
                # FIXME: hardcode here
                caption_labels=(language.clone() if (isinstance(rgb, torch.Tensor) and rgb.shape[1] == 1 and
                                                     isinstance(language, torch.Tensor)) else None),
                caption_mask=(text_mask.clone() if (isinstance(rgb, torch.Tensor) and rgb.shape[1] == 1 and
                                                    isinstance(text_mask, torch.Tensor)) else None),
                vision_gripper=hand_rgb,
                fwd_rgb_labels=fwd_rgb_chunck,
                fwd_hand_rgb_labels=fwd_hand_rgb_chunck,
                fwd_mask=fwd_mask,
                instr_and_action_ids=instr_and_action_ids,
                instr_and_action_labels=instr_and_action_labels,
                instr_and_action_mask=instr_and_action_mask,
                raw_text=raw_text,
                rel_state=rel_state,
            )

            return prediction

    @staticmethod
    def convert_old_state_dict(state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_k = k.replace("module.", "")
            else:
                new_k = k

            if not new_k.startswith("model."):
                new_k = "model." + new_k

            new_state_dict[new_k] = state_dict[k]
        return new_state_dict

    def get_grouped_params(self, model):
        return [{
            "params": [p for n, p in model.named_parameters() if p.requires_grad],
            "weight_decay": self.configs["weight_decay"],
        }]
