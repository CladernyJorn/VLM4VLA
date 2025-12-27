from sympy.logic.boolalg import true
import torch
from vlm4vla.model.backbone.base_backbone import BaseRoboVLM, load_config, deep_update
from einops import rearrange, repeat
import json
import os, sys, copy
import numpy as np
from typing import Optional, Tuple, List

import torch
from torch import nn

from vlm4vla.utils.model_utils import update_tokenizer
from vlm4vla.model.backbone.pi0_model.vla.pizero import PiZero
from vlm4vla.model.backbone.pi0_model.vla.processing import VLAProcessor
from omegaconf import OmegaConf, open_dict
from transformers.models.auto.processing_auto import AutoProcessor
from transformers.models.paligemma.modeling_paligemma import PaliGemmaForConditionalGeneration


class Pi0Paligemma(nn.Module):

    def __init__(
        self,
        configs,
        train_setup_configs,
        act_encoder_configs=None,
        act_head_configs=None,
        fwd_head_configs=None,
        window_size=None,
        use_obs_queries=True,
        use_act_queries=True,
        use_hand_rgb=False,
        use_pixel_loss=True,
        use_mim_obs_loss=False,
        use_time_causal_attn=True,
        vision_masked_ratio=0.9,
        use_tube_mask=False,
        fwd_pred_next_n=1,
        use_vision_resampler=False,
        vision_resampler_configs=None,
        use_clip_norm=False,
        use_state=False,
        **kwargs,
    ):
        super().__init__()
        self.window_size = window_size  #8
        self.use_obs_queries = use_obs_queries  #True
        self.use_act_queries = use_act_queries  #False
        self.use_hand_rgb = use_hand_rgb  #False
        self.use_pixel_loss = use_pixel_loss  #True
        self.use_mim_obs_loss = use_mim_obs_loss  #False
        self.use_time_causal_attn = use_time_causal_attn  #True, args in config not used
        self.use_clip_norm = use_clip_norm  #False
        self.vision_masked_ratio = vision_masked_ratio  #0.9
        self.use_tube_mask = use_tube_mask  #False
        self.use_state = use_state  #False
        self.fwd_pred_next_n = fwd_pred_next_n  #10

        self.kwargs = kwargs
        self.configs = configs
        self.model_name = configs["model"]
        # self.model_config = json.load(
        #     open(
        #         os.path.join(self.configs["vlm"]["pretrained_model_name_or_path"], "config.json"),
        #         "r",
        #     ))

        self.train_setup_configs = train_setup_configs
        self.act_encoder_configs = act_encoder_configs  # None
        self.act_head_configs = act_head_configs
        self.fwd_head_configs = fwd_head_configs  # None
        self.vision_resampler_configs = vision_resampler_configs  # not None, but use_vision_resampler is False, so not used

        # self.tokenizer, self.backbone = self._init_backbone()
        # import pdb;pdb.set_trace()
        # self.tokenizer = update_tokenizer(self.tokenizer, self.configs["tokenizer"])  # did nothing to qwen25vl
        # if self.train_setup_configs.get("reinit", False):  # False
        #     initialize_param(self.backbone)
        # self.act_head, self.fwd_head, self.clip_norm_head = self._init_heads()

        # if self.act_head_configs is not None:
        #     self.action_space = self.act_head_configs.get("action_space", "continuous")
        #     if self.action_space == "discrete":
        #         self.action_tokenizer = ActionTokenizer(
        #             self.tokenizer,
        #             bins=self.act_head_configs["n_bin"],
        #             min_action=self.act_head_configs["min_action"],
        #             max_action=self.act_head_configs["max_action"],
        #         )

        #     if self.action_space == "continuous":
        #         self.action_token = nn.Parameter(torch.zeros(self.hidden_size))
        #         self.action_token.requires_grad_(True)

        ### setup vision tower and configs

        # if self.use_state:
        #     # Embedding functions for states
        #     state_dim = 7
        #     self.embed_arm_state = torch.nn.Linear(state_dim - 1, self.hidden_size)
        #     self.embed_gripper_state = torch.nn.Embedding(2, self.hidden_size)  # one-hot gripper state
        #     self.embed_state = torch.nn.Linear(2 * self.hidden_size, self.hidden_size)

        # load pi0 configs from yaml file
        pi0_cfg = OmegaConf.load(self.configs["pi0_cfg"])
        pi0_cfg.horizon_steps = self.fwd_pred_next_n
        # pi0_cfg.cond_steps = self.window_size
        if not self.use_state:
            pi0_cfg.cond_steps = 0  # no proprio tokens
            pi0_cfg.mixture.proprio.cache = False
        else:
            pi0_cfg.cond_steps = self.window_size  # use proprio tokens
        pi0_cfg.pretrained_model_path = self.configs["model_path"]
        # self.configs["pi0_cfg"] = pi0_cfg
        self.pi0_cfg = pi0_cfg
        self.model = PiZero(self.pi0_cfg, use_ddp=True)
        if not self.use_state:
            self.model.num_proprio_tokens = 0
        self.dtype = torch.bfloat16 if pi0_cfg.get("use_bf16", True) else torch.float32
        # if cfg.resume_checkpoint_path:
        #     self.load_checkpoint(cfg.resume_checkpoint_path)
        if pi0_cfg.load_pretrained_weights:
            self.model.load_pretrained_weights()
        self.model.tie_action_proprio_weights()
        self.model.freeze_unused_weights()
        # self.model.to("cuda")
        # 确保模型参数类型一致性，避免混合精度类型错误
        self.model = self.model.to(self.dtype)
        # 将模型移动到与self.dtype相同的设备
        # device = next(self.model.parameters()).device
        # self.model.to(device)
        # self.device = device
        # print(self.device) # cpu

        ########### Input processing ###########

        # flow matching timestep sampling
        self.flow_sampling = pi0_cfg.get("flow_sampling", "beta")
        assert self.flow_sampling in [
            "uniform",
            "beta",
        ], f"Invalid flow matching timestep sampling mode: {self.flow_sampling}"
        if self.flow_sampling == "beta":
            flow_alpha = pi0_cfg.get("flow_alpha", 1.5)
            flow_beta = pi0_cfg.get("flow_beta", 1)
            self.flow_t_max = 1 - pi0_cfg.get("flow_sig_min", 0.001)
            self.flow_beta_dist = torch.distributions.Beta(flow_alpha, flow_beta)

        self.processor = AutoProcessor.from_pretrained(pi0_cfg.pretrained_model_path, padding_side="right")
        self.tokenizer = self.processor.tokenizer
        self.image_token = "<image>"
        tokens_to_add = {"additional_special_tokens": [self.image_token]}
        self.tokenizer.add_special_tokens(tokens_to_add)
        EXTRA_TOKENS = [f"<loc{i:04d}>" for i in range(1024)
                       ]  # These tokens are used for object detection (bounding boxes)
        EXTRA_TOKENS += [f"<seg{i:03d}>" for i in range(128)]  # These tokens are used for object segmentation
        self.tokenizer.add_tokens(EXTRA_TOKENS)
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
        # We will add the BOS and EOS tokens ourselves
        self.tokenizer.add_bos_token = False
        self.tokenizer.add_eos_token = False

    def preprocess_inputs(self, lang_x, vision_x, attention_mask, rel_state, action_labels, split_mask, sample_fm_time):
        # build causal mask and position ids for action
        if action_labels is not None:
            action_labels = torch.cat([action_labels[0], action_labels[1].unsqueeze(-1)], dim=-1).squeeze(1)
        causal_mask, vlm_position_ids, proprio_position_ids, action_position_ids = (
            self.model.build_causal_mask_and_position_ids(attention_mask, self.dtype))

        inputs = {
            "input_ids": lang_x,
            "pixel_values": vision_x.to(self.dtype).squeeze(1),
            "vlm_position_ids": vlm_position_ids,
            "proprio_position_ids": proprio_position_ids,
            "action_position_ids": action_position_ids,
            "proprios": rel_state.to(self.dtype).squeeze(1) if self.use_state else None,
            "actions": action_labels.to(self.dtype) if action_labels is not None else None,
        }
        if split_mask:
            image_text_proprio_mask, action_mask = (self.model.split_full_mask_into_submasks(causal_mask))
            inputs["image_text_proprio_mask"] = image_text_proprio_mask
            inputs["action_mask"] = action_mask
        else:
            inputs["causal_mask"] = causal_mask

        # sample flow matching timesteps
        if sample_fm_time:
            inputs["t"] = self.sample_fm_time(lang_x.shape[0]).to(self.dtype)
        for k, v in inputs.items():
            # qwen has input_ids, attention_mask, pixel_values, image_grid_thw
            if v is not None:
                inputs[k] = v.cuda()
        # inputs = {k: v.to(self.device) if v is not None else None for k, v in inputs.items()}
        return inputs

    def forward_continuous(
        self,
        vision_x: torch.Tensor,  # rgb
        lang_x: torch.Tensor,  # language
        attention_mask: torch.Tensor = None,  # text mask
        action_labels: Tuple[torch.Tensor, torch.Tensor] = None,  # arm_action_chunck, gripper_action_chunck
        raw_text=None,
        rel_state=None,  # not used (not transfered from forward_action)
        mode="train",
        **kwargs,
    ):
        # used in input:
        # vision_x, lang_x, attention_mask, action_labels, action_mask, vision_gripper,raw_text
        # dataset中的预处理全部用的paligemma的(由configs["model"]决定)
        loss = {}
        assert vision_x is not None
        bs, seq_len = vision_x.shape[:2]  # (4, 8), seq_len 就是window size

        # print("lang_x:", lang_x.shape)
        if mode == "train":
            inputs = self.preprocess_inputs(
                lang_x, vision_x, attention_mask, rel_state, action_labels, split_mask=False, sample_fm_time=True)
            with torch.autocast(device_type="cuda", dtype=self.dtype, enabled=True):
                output = self.model(**inputs)
            action_loss = {"loss_arm": output}
            self._update_loss(loss, action_loss, "act")
            loss = self._format_loss(loss)
        else:
            # val or inference
            inputs = self.preprocess_inputs(
                lang_x, vision_x, attention_mask, rel_state, action_labels, split_mask=True, sample_fm_time=False)
            gt_action = inputs.pop("actions")
            with torch.autocast(device_type="cuda", dtype=self.dtype, enabled=True):
                # inputs = {k: v.to("cuda") if v is not None else None for k, v in inputs.items()}
                action_logits = self.model.infer_action(**inputs)
            # print(action_logits.shape) # 16,4,7
            # print(gt_action.shape) # 16,4,7
            if mode == "val":
                action_loss = self.calculate_val_loss(action_logits, gt_action)
                eval_accuracy = get_action_accuracy(gt_action, action_logits)
                eval_l1_loss = {"eval_l1": torch.nn.functional.l1_loss(action_logits, gt_action)}
                self._update_loss(loss, action_loss, "act")
                self._update_loss(loss, eval_accuracy, "act")
                self._update_loss(loss, eval_l1_loss, "act")
                loss = self._format_loss(loss)
            # return loss
            elif mode == "inference":
                action = action_logits[..., :6]
                gripper = action_logits[..., -1]
                return action, gripper

        return loss

    def calculate_val_loss(self, pred_action, labels, attention_mask=None):
        """
        pred_action_logits: [bs, seq_len, chunck_size, 7], 1-6 refers to ee pose, 7 refers to gripper open/close
        lables: (pose gt [bs, seq_len, chunck_size, 6], gripper gt [bs, seq_len, chunck_size])
        attention_mask: [bs, seq_len, chunck_size]
        """

        pose_loss = torch.mean((pred_action[..., :6] - labels[..., :6])**2)
        gripper_loss = torch.mean((pred_action[..., -1] - labels[..., -1])**2)
        return {
            "loss_arm": pose_loss,
            "loss_gripper": gripper_loss,
            # "acc_gripper": acc_gripper_act.item(),
        }

    def forward(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,  # not used
        use_cached_vision_x: bool = False,  # TODO: Do we need this? If not we can remove it from here # not used
        action_labels: Tuple[torch.Tensor, torch.Tensor] = None,
        action_mask: torch.Tensor = None,
        caption_labels: torch.Tensor = None,  # not used
        caption_mask: torch.Tensor = None,  # not used
        past_key_values=None,  # not used
        use_cache: bool = False,  # not used
        vision_gripper=None,
        fwd_rgb_labels: torch.Tensor = None,  # not used
        fwd_hand_rgb_labels: torch.Tensor = None,  # not used
        fwd_mask: torch.Tensor = None,  # not used
        instr_and_action_ids=None,  # not used
        instr_and_action_labels=None,  # not used
        instr_and_action_mask=None,  # not used
        raw_text=None,
        data_source=[],
        **kwargs,
    ):
        loss = {}

        tmp_loss = self.forward_action(
            vision_x=vision_x,
            lang_x=lang_x,
            attention_mask=attention_mask,
            action_labels=action_labels,
            action_mask=action_mask,
            caption_labels=caption_labels,
            caption_mask=caption_mask,
            vision_gripper=vision_gripper,
            fwd_rgb_labels=fwd_rgb_labels,
            fwd_hand_rgb_labels=fwd_hand_rgb_labels,
            fwd_mask=fwd_mask,
            instr_and_action_ids=instr_and_action_ids,
            instr_and_action_labels=instr_and_action_labels,
            instr_and_action_mask=instr_and_action_mask,
            raw_text=raw_text,
            rel_state=kwargs.get("rel_state", None),
            mode=kwargs.get("mode", "train"),
        )
        loss = self._update_loss(loss, tmp_loss)

        return loss

    def forward_action(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        use_cached_vision_x: bool = False,  # TODO: Do we need this? If not we can remove it from here
        action_labels: Tuple[torch.Tensor, torch.Tensor] = None,
        action_mask: torch.Tensor = None,
        caption_labels: torch.Tensor = None,  # not used
        caption_mask: torch.Tensor = None,  # not used
        past_key_values=None,
        use_cache: bool = False,
        vision_gripper=None,
        fwd_rgb_labels: torch.Tensor = None,  # not used
        fwd_hand_rgb_labels: torch.Tensor = None,  # not used
        fwd_mask: torch.Tensor = None,  # not used
        instr_and_action_ids=None,  # not used
        instr_and_action_labels=None,  # not used
        instr_and_action_mask=None,  # not used
        raw_text=None,
        rel_state=None,
        mode="train",
        **kwargs,
    ):
        action_space = self.act_head_configs.get("action_space", "continuous")
        ### discard the latter visual observation is with_history is False
        ### while we can maintain the multi-step action (chunk) prediction

        return self.forward_continuous(
            vision_x=vision_x,
            lang_x=lang_x,
            attention_mask=attention_mask,
            action_labels=action_labels,
            action_mask=action_mask,
            caption_labels=caption_labels,  # not used
            caption_mask=caption_mask,  # not used
            vision_gripper=vision_gripper,
            fwd_rgb_labels=fwd_rgb_labels,  # not used
            fwd_hand_rgb_labels=fwd_hand_rgb_labels,  # not used
            fwd_mask=fwd_mask,  # not used
            instr_and_action_ids=instr_and_action_ids,  # not used
            instr_and_action_labels=instr_and_action_labels,  # not used
            instr_and_action_mask=instr_and_action_mask,  # not used
            raw_text=raw_text,
            rel_state=rel_state,
            mode=mode,
        )

    def inference(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        use_cached_vision_x: bool = False,  # TODO: Do we need this? If not we can remove it from here
        action_labels: Tuple[torch.Tensor, torch.Tensor] = None,
        action_mask: torch.Tensor = None,
        caption_labels: torch.Tensor = None,
        caption_mask: torch.Tensor = None,
        past_key_values=None,
        use_cache: bool = False,
        vision_gripper=None,
        **kwargs,
    ):
        prediction = {}

        assert vision_x is not None
        bs, seq_len = vision_x.shape[:2]
        action_space = self.act_head_configs.get("action_space", "continuous")
        if self.train_setup_configs["predict_action"]:
            prediction["action"] = self.forward_continuous(
                vision_x,
                lang_x,
                attention_mask,
                vision_gripper=vision_gripper,
                mode="inference",
            )

        return prediction

    @staticmethod
    def _update_loss(loss, new_loss, suffix=None):
        """
        use new_loss to update loss.
            * if suffix is not None, the key from new_loss will be reformatted as: key|suffix
            * otherwise, if the key from new_loss is not in loss, it will be directly used: key
            * otherwise, the key from the new_loss will be reformatted as: key|index, where index is
                searched from 0->+inf so that key|index is not in loss.

        """

        def get_key(k, d):
            if suffix is not None:
                new_k = f"{k}_{suffix}"
                assert new_k not in d
                return new_k

            ind = 0
            while True:
                if ind == 0:
                    new_k = k
                else:
                    new_k = f"{k}_{ind}"
                if new_k not in d:
                    return new_k
                ind += 1

        for k in new_loss:
            new_k = get_key(k, loss)
            loss[new_k] = new_loss[k]

        return loss

    def _format_loss(self, loss):
        # for visualization and loss backward in pytorch lightning
        _loss = 0
        _keys = list(loss.keys())

        for k in _keys:
            if "loss" in k:
                _loss += loss[k]

        loss["loss"] = _loss
        return loss

    def sample_fm_time(self, bsz: int) -> torch.FloatTensor:
        if self.flow_sampling == "uniform":  # uniform between 0 and 1
            """https://github.com/gle-bellier/flow-matching/blob/main/Flow_Matching.ipynb"""
            eps = 1e-5
            t = (torch.rand(1) + torch.arange(bsz) / bsz) % (1 - eps)
        elif self.flow_sampling == "beta":  # from pi0 paper
            z = self.flow_beta_dist.sample((bsz,))
            t = self.flow_t_max * (1 - z)  # flip and shift
        return t

    @property
    def image_processor(self):
        return self.processor

    @property
    def hidden_size(self):
        return self.model.config.text_config.hidden_size

    @property
    def word_embedding(self):
        return self.model.language_model.model.embed_tokens

    @property
    def text_tower(self):
        return self.model.language_model.model

    @property
    def vision_tower(self):
        return self.model.vision_tower

    # @property
    # def model(self):
    #     return self.backbone


def get_action_accuracy(
    gt: torch.FloatTensor,  # [Batch_Size, Horizon, Action_Dim]
    pred: torch.FloatTensor,
    thresholds=[0.5, 0.3, 0.2, 0.1, 0.05],
) -> torch.FloatTensor:
    device = gt.device
    diff = torch.abs(gt - pred).reshape(-1, gt.shape[-1])

    # get the percentage of diff lower than threshold for all action dimensions
    accuracies = {}
    for idx, threshold in enumerate(thresholds):
        accuracy = torch.mean((torch.mean((diff < threshold).float(), dim=1) >= 1.0).float())
        accuracies["eval_acc_thres " + str(threshold)] = accuracy
    return accuracies


if __name__ == "__main__":
    configs = load_config(
        "configs/finetune_paligemma_cont-lstm-post_full-ft_text_vision_wd=0_hist=8_act=10_aug-shift_act-norm_lr-2e-5.json"
    )
    use_hand_rgb = False  # True
    model = RoboPaligemma(
        configs=configs,
        train_setup_configs=configs["train_setup"],
        fwd_head_configs=None,
        window_size=configs["window_size"],
        use_hand_rgb=use_hand_rgb,
        act_head_configs=configs["act_head"],
        fwd_pred_next_n=configs["fwd_pred_next_n"],
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {total_params / 1000000:.2f}M")
    # import pdb; pdb.set_trace()
    bs, seq_len = 2, 2
    device = "cuda:0"
    # device = 'cpu'
    vision_x = torch.zeros((bs, seq_len, 3, 224, 224), dtype=torch.float16).to(device)
    vision_gripper = torch.zeros((bs, seq_len, 3, 224, 224), dtype=torch.float16).to(device)
    lang_x = torch.ones((bs, 10), dtype=torch.long).to(device) * 100
    attention_mask = torch.ones((bs, 10)).bool().to(device)
    action_lables = (
        torch.randn(bs, seq_len, configs["fwd_pred_next_n"], 6).to(device),
        torch.zeros(bs, seq_len, configs["fwd_pred_next_n"]).to(device),
    )
    model = model.to(device).to(torch.float16)
    rel_state = torch.randn(bs, seq_len, 7).to(device)
    rel_state[..., -1] = 0
    test_res = model(
        vision_x,
        lang_x,
        attention_mask=attention_mask,
        position_ids=None,
        use_cached_vision_x=False,
        action_labels=action_lables,
        action_mask=None,
        caption_labels=None,
        caption_mask=None,
        past_key_values=None,
        use_cache=False,
        vision_gripper=vision_gripper,
        fwd_rgb_labels=None,
        fwd_hand_rgb_labels=None,
        fwd_mask=None,
        data_source=["calvin_action"],
        rel_state=rel_state,
    )

    print(test_res)
