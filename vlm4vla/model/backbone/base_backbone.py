from einops import rearrange, repeat
import json
import os, sys, copy
import numpy as np
from typing import Optional, Tuple, List

import torch
from torch import nn

from vlm4vla.utils.model_utils import update_tokenizer
from vlm4vla.model.vlm_builder import build_vlm


def initialize_param(model):
    with torch.no_grad():
        for m in model.children():
            if hasattr(m, "weight") and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    m.bias.fill_(0)
            else:
                initialize_param(m)


class BaseRoboVLM(nn.Module):

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
        self.model_config = json.load(
            open(
                os.path.join(self.configs["vlm"]["pretrained_model_name_or_path"], "config.json"),
                "r",
            ))

        self.train_setup_configs = train_setup_configs
        self.act_encoder_configs = act_encoder_configs  # None
        self.act_head_configs = act_head_configs
        self.fwd_head_configs = fwd_head_configs  # None
        self.vision_resampler_configs = vision_resampler_configs  # not None, but use_vision_resampler is False, so not used

        self.tokenizer, self.backbone = self._init_backbone()
        if self.train_setup_configs.get("gradient_checkpointing", True):
            self.backbone.gradient_checkpointing_enable()

        self.tokenizer = update_tokenizer(self.tokenizer, self.configs["tokenizer"])  # did nothing to qwen25vl
        if self.train_setup_configs.get("reinit", False):  # False
            initialize_param(self.backbone)
        self.act_head, self.fwd_head, self.clip_norm_head = self._init_heads()

        if self.act_head_configs is not None:
            self.action_space = self.act_head_configs.get("action_space", "continuous")
        assert self.action_space == "continuous"

        self.action_token = nn.Parameter(torch.zeros(self.hidden_size))
        self.action_token.requires_grad_(True)

        if self.fwd_head_configs is not None:
            self.image_latent_num = self.fwd_head_configs.get("image_latent_num", 8)
            self.pred_image = True
            self.pred_hand_image = self.fwd_head_configs.get("pred_hand_image", False)

            global_frame_num = self.fwd_head_configs.get("global_frame_num", -1)
            if global_frame_num != -1:
                predict_image_num = global_frame_num - 1
            else:
                predict_image_num = self.fwd_pred_next_n

            self.static_image_tokens = nn.Parameter(
                torch.zeros(self.image_latent_num * predict_image_num, self.hidden_size))
            self.static_image_tokens.requires_grad_(True)
            if self.pred_hand_image:
                self.hand_image_tokens = nn.Parameter(
                    torch.zeros(self.image_latent_num * predict_image_num, self.hidden_size))
                self.hand_image_tokens.requires_grad_(True)
        else:
            self.pred_image = False

        if self.use_state:
            # Embedding functions for states
            state_dim = 7
            self.embed_arm_state = torch.nn.Linear(state_dim - 1, self.hidden_size)
            self.embed_gripper_state = torch.nn.Embedding(2, self.hidden_size)  # one-hot gripper state
            self.embed_state = torch.nn.Linear(2 * self.hidden_size, self.hidden_size)

        self._trainable_params_setup()

    def encode_state(self, state):
        arm_state_embeddings = self.embed_arm_state(state[..., :6])
        gripper_state_embeddings = self.embed_gripper_state(state[..., [-1]]).long()
        state_embeddings = torch.cat((arm_state_embeddings, gripper_state_embeddings), dim=2)
        state_embeddings = self.embed_state(state_embeddings)  # (b, l, h)
        return state_embeddings

    def model_encode_images(self, images):
        raise NotImplementedError

    def encode_images(self, images, image_sizes=None):
        # input: images: list of b,c,h,w or b,t,c,h,w
        # output: image_features: b, t, n, d

        if images.ndim == 4:
            images = images.unsqueeze(1)

        bs, seq_len = images.shape[:2]

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.model_encode_images(concat_images)

            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]

        else:
            image_features = self.model_encode_images(images)

        image_features = torch.stack(image_features, dim=0).view(bs, seq_len, -1, image_features[0].shape[-1])

        return image_features

    def _init_backbone(self):
        tokenizer, model = build_vlm(self.configs["vlm"], self.configs["tokenizer"])
        if "Processor" in self.configs["tokenizer"]["type"]:
            self.processor = tokenizer
            self.tokenizer = self.processor.tokenizer
        else:
            self.tokenizer = tokenizer
        return self.tokenizer, model

    @property
    def hidden_size(self):
        raise NotImplementedError

    @property
    def word_embedding(self):
        raise NotImplementedError

    @property
    def vision_tower(self):
        raise NotImplementedError

    @property
    def text_tower(self):
        raise NotImplementedError

    @property
    def model(self):
        raise NotImplementedError

    @property
    def start_image_token_id(self):
        return None

    @property
    def end_image_token_id(self):
        return None

    @property
    def image_processor(self):
        # return None
        import torchvision.transforms as T

        clip_mean = (0.48145466, 0.4578275, 0.40821073)
        clip_std = (0.26862954, 0.26130258, 0.27577711)
        image_preprocess = T.Compose([
            T.Resize(
                (
                    self.configs.get("image_size", 224),
                    self.configs.get("image_size", 224),
                ),
                interpolation=T.InterpolationMode.BICUBIC,
            ),
            T.Lambda(lambda img: img.convert("RGB")),
            T.ToTensor(),
            T.Normalize(clip_mean, clip_std),
        ])
        return image_preprocess

    def merge_multi_modal_input(
        self,
        input_embeds: torch.Tensor,
        multimodal_feats: torch.Tensor = None,
        labels: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        is_image=True,
        insert_idx=1,
        fill_zero=False,
    ):
        # if is_image, the vision_x needs to be processed by self.encode_images
        # otherwise merge
        bs = input_embeds.shape[0]

        if is_image:
            rgb_feats = self.encode_images(multimodal_feats)

            if self.start_image_token_id is not None:
                image_start_embed = (
                    self.word_embedding(self.start_image_token_id.to(
                        self.model.device)).unsqueeze(0).unsqueeze(0).repeat(*rgb_feats.shape[:2], 1, 1))

                if self.end_image_token_id is None:
                    end_image_token_id = self.start_image_token_id + 1
                else:
                    end_image_token_id = self.end_image_token_id
                image_end_embed = (
                    self.word_embedding(end_image_token_id.to(self.model.device)).unsqueeze(0).unsqueeze(0).repeat(
                        *rgb_feats.shape[:2], 1, 1))

                rgb_feats = torch.cat([image_start_embed, rgb_feats, image_end_embed], dim=2)

            rgb_feats = rearrange(
                rgb_feats,
                "b l n d -> b (l n) d")  # flatten seq_len and n_tok_per_img dim, (4*8,1,n,d)->(4*8,n,d) (32,256,2304)

        else:
            rgb_feats = multimodal_feats

        added_seq_len = rgb_feats.shape[1]

        multimodal_embeds = torch.cat(
            [input_embeds[:, :insert_idx], rgb_feats, input_embeds[:, insert_idx:]],
            dim=1,
        )

        insert_mask = (
            torch.cat(
                [
                    torch.zeros(input_embeds[:, :insert_idx].shape[:2]),
                    torch.ones(rgb_feats.shape[:2]),
                    torch.zeros(input_embeds[:, insert_idx:].shape[:2]),
                ],
                dim=1,
            ).bool().to(multimodal_embeds.device))

        mutlimodal_labels = None
        if labels is not None:
            mutlimodal_labels = torch.full((bs, added_seq_len), -100, dtype=labels.dtype, device=labels.device)
            mutlimodal_labels = self.cat_multi_modal_input(labels, mutlimodal_labels, insert_idx, attention_mask)
            if is_image:
                if self.start_image_token_id is not None:
                    mutlimodal_labels[:, 0] = self.start_image_token_id
                    mutlimodal_labels[:, multimodal_feats.shape[1] + 1] = end_image_token_id

        multimodal_attention_mask = None
        if attention_mask is not None:
            val = False if fill_zero else True
            multimodal_attention_mask = torch.full(
                (bs, added_seq_len),
                val,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            multimodal_attention_mask = self.cat_multi_modal_input(attention_mask, multimodal_attention_mask,
                                                                   insert_idx, attention_mask)

        return (
            multimodal_embeds,
            mutlimodal_labels,
            multimodal_attention_mask,
            insert_mask,
        )

    def _init_heads(self):
        action_head = None
        if self.act_head_configs is not None:
            import vlm4vla.model.policy_head as action_heads

            _kwargs = copy.deepcopy(self.act_head_configs)
            _kwargs.update(
                dict(  # hidden_size=self.hidden_size,
                    tokenizer=self.tokenizer,
                    in_features=self.hidden_size,
                    fwd_pred_next_n=self.fwd_pred_next_n,
                    window_size=self.window_size,
                    n_bin=self.act_head_configs.get("n_bin", 256),
                    min_action=self.act_head_configs.get("min_action", -1),
                    max_action=self.act_head_configs.get("max_action", 1),
                ))
            _cls = getattr(action_heads, _kwargs.pop("type"))
            self.latent_num = self.act_head_configs.get("latent", 1)
            action_head = _cls(**_kwargs)

        return action_head, None, None
    
    def cat_multi_modal_input(
        self,
        input_embeds: torch.Tensor,
        multimodal_embeds: Optional[torch.Tensor] = None,
        insert_idx: int = 0,
        masks: Optional[torch.Tensor] = None,
    ):
        # concat multi-modal inputs
        if insert_idx >= 0:
            return torch.cat(
                (
                    input_embeds[:, :insert_idx],
                    multimodal_embeds,
                    input_embeds[:, insert_idx:],
                ),
                dim=1,
            )
        elif insert_idx == -1 and masks is not None:
            new_embed_list = []
            for mask, input_embed, multimodal_embed in zip(masks, input_embeds, multimodal_embeds):
                # the concat index is up to mask first False index
                # concat_idx = (mask == False).nonzero()[0].item()
                indexs = (mask == False).nonzero()
                insert_idx = indexs[0].item() if len(indexs) > 0 else len(mask)
                new_embed = torch.cat(
                    (
                        input_embed[:insert_idx],
                        multimodal_embed,
                        input_embed[insert_idx:],
                    ),
                    dim=0,
                )
                new_embed_list.append(new_embed)
            return torch.stack(new_embed_list, dim=0)
        else:
            raise Exception(
                "insert_idx should be -1 or >= 0, and if you want to insert as last(-1), you should provide masks")

    def _trainable_params_setup(self):
        model = self.model
        compute_dtype = (
            torch.float32
        )  # (torch.float16 if self.train_setup_configs['precision'] == 'fp16' else (torch.bfloat16 if self.train_setup_configs['precision'] == 'bf16' else torch.float32))

        model.config.use_cache = False
        # print(self.train_setup_configs)
        if self.train_setup_configs["freeze_backbone"]:  # false
            model.requires_grad_(False)
        else:
            if self.train_setup_configs.get("train_decoder_layers", -1) == -1:
                model.requires_grad_(True)
            else:
                model.requires_grad_(False)
                if hasattr(self.text_tower, "layers"):
                    for layer in self.text_tower.layers[-self.train_setup_configs["train_decoder_layers"]:]:
                        layer.requires_grad_(True)
                elif hasattr(self.text_tower, "blocks"):
                    for layer in self.text_tower.blocks[-self.train_setup_configs["train_decoder_layers"]:]:
                        layer.requires_grad_(True)

        if self.train_setup_configs.get("train_vision", False):  # True
            self.vision_tower.requires_grad_(True)
        else:
            self.vision_tower.requires_grad_(False)

        if hasattr(model, "enable_input_require_grads"):
            # print("enable_input_require_grads") for qwen2.5-vl, has this function
            model.enable_input_require_grads()
            model.gradient_checkpointing = True
            model.training = True
        else:
            # print("make_inputs_require_grad") # not used in qwen2.5-vl
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            self.word_embedding.register_forward_hook(make_inputs_require_grad)

        # import pdb; pdb.set_trace()
        if self.train_setup_configs.get("train_text_embedding", False):  # True
            self.word_embedding.requires_grad_(True)
        else:
            self.word_embedding.requires_grad_(False)

        if self.act_head is not None:
            self.act_head.requires_grad_(True)
        # print({k for k, v in self.named_parameters() if v.requires_grad})

    def _forward_action_head(
        self,
        action_tokens: torch.Tensor,
        action_labels: Tuple[torch.Tensor, torch.Tensor] = None,
        action_mask: torch.Tensor = None,
        **kwargs,
    ):
        # action_tokens = get_target_modal_tokens(output_hs, self._action_mask(output_hs))
        action = self.act_head(action_tokens, actions=action_labels, action_masks=action_mask, **kwargs)

        action_loss = None
        if action_labels is not None:
            action, action_labels, action_mask = self.act_head.get_labels(
                action, action_labels, action_mask, tok_seq=action_tokens, **kwargs)
            action_loss = self.act_head.loss(action, action_labels, action_mask)

        return action, action_loss

    def _format_loss(self, loss):
        # for visualization and loss backward in pytorch lightning
        _loss = 0
        _keys = list(loss.keys())

        for k in _keys:
            if "loss" in k:
                _loss += loss[k]

        loss["loss"] = _loss
        return loss

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

    def forward_continuous(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,  # not used (not transfered from forward_action)
        action_labels: Tuple[torch.Tensor, Optional[torch.Tensor]] = None,
        action_mask: torch.Tensor = None,
        vision_gripper=None,
        raw_text=None,
        rel_state=None,  # not used (not transfered from forward_action)
        mode="train",
        **kwargs,
    ):
        # used in input:
        # vision_x, lang_x, attention_mask, action_labels, action_mask, vision_gripper,raw_text
        loss = {}
        assert vision_x is not None
        bs, seq_len = vision_x.shape[:2]  # (4, 8), seq_len 就是window size

        history_type = self.act_head_configs.get("history_type", "post")
        assert history_type == "post"

        vision_x = vision_x.reshape(bs * seq_len, *vision_x.shape[2:]).unsqueeze(1)  # (4*8,1, 3, 224, 224)
        # lang_x = lang_x.repeat(seq_len, 1)
        # attention_mask = attention_mask.repeat(seq_len, 1)
        lang_x = lang_x.unsqueeze(1).repeat(1, seq_len, 1).flatten(0, 1)  # original shape: (4, 12) -> (4*8, 12)
        attention_mask = (attention_mask.unsqueeze(1).repeat(1, seq_len, 1).flatten(0, 1))
        if vision_gripper is not None:
            vision_gripper = vision_gripper.reshape(bs * seq_len, *vision_gripper.shape[2:]).unsqueeze(1)

        input_embeds = self.word_embedding(lang_x)  # (4*8, 12, 1024)
        # get <bos> & <eos> offset
        lang_size = (
            lang_x.shape[-1] - int(self.tokenizer.eos_token is not None) - int(self.tokenizer.bos_token is not None))

        (
            multimodal_embeds,
            mutlimodal_labels,
            multimodal_attention_mask,
            _,
        ) = self.merge_multi_modal_input(
            input_embeds,
            vision_x,
            labels=None,
            attention_mask=attention_mask,
            # insert_idx=bos_offset,
            insert_idx=0,  # fixed for paligemma, where |<img_token><bos><text>\n<pad>|
        )

        # print("after insert image, insert mask", _)
        # print("after insert image, multimodal attention mask", multimodal_attention_mask)
        if vision_gripper is not None:
            (
                multimodal_embeds,
                mutlimodal_labels,
                multimodal_attention_mask,
                _,
            ) = self.merge_multi_modal_input(
                multimodal_embeds,
                vision_gripper,
                mutlimodal_labels,
                multimodal_attention_mask,
                # insert_idx=bos_offset,
                insert_idx=0,  # fixed for paligemma, where |<img_token><bos><text>\n<pad>|
            )

        if rel_state is not None and self.use_state:
            print("Using state!!!!!!!!!!!!")
            # insert_idx = multimodal_embeds.shape[1] - int(self.tokenizer.eos_token is not None)  # insert at last
            state_token = self.encode_state(rel_state)  # bs, seq_len, d
            state_token = state_token.reshape(bs * seq_len, state_token.shape[-1]).unsqueeze(1)  # bs*seq_len, 1, d
            (
                multimodal_embeds,
                mutlimodal_labels,
                multimodal_attention_mask,
                action_token_mask,
            ) = self.merge_multi_modal_input(
                multimodal_embeds,
                state_token,
                mutlimodal_labels,
                multimodal_attention_mask,
                is_image=False,
                # insert_idx=insert_idx,
                insert_idx=multimodal_embeds.shape[1],
                # fixed for paligemma, where |<img_token><bos><text>\n<pad>|, after insert <state>: |<img_token><bos><text>\n<pad><state>|
                fill_zero=self.act_head_configs.get("fill_zero", False),
            )

            # insert_idx = multimodal_embeds.shape[1] - int(self.tokenizer.eos_token is not None)  # insert at last

        action_tokens = repeat(
            self.action_token,
            "d -> b n d",
            b=multimodal_embeds.shape[0],
            n=self.latent_num,
        )

        (
            multimodal_embeds,
            mutlimodal_labels,
            multimodal_attention_mask,
            action_token_mask,
        ) = self.merge_multi_modal_input(
            multimodal_embeds,
            action_tokens,
            mutlimodal_labels,
            multimodal_attention_mask,
            is_image=False,
            # insert_idx=insert_idx,
            insert_idx=multimodal_embeds.shape[1],
            # fixed for paligemma, where |<img_token><bos><text>\n<pad>|, after insert <action>: |<img_token><bos><text>\n<pad><action>|
            fill_zero=self.act_head_configs.get("fill_zero", False),
        )
        # print("after insert action, insert mask", action_token_mask)
        # print("after insert action, multimodal attention mask", multimodal_attention_mask)

        if history_type == "pre":
            multimodal_embeds = rearrange(
                multimodal_embeds, "(b l) n d -> b (l n) d",
                l=seq_len)  # original shape: (4*8, 271, 2304) -> (4, 8*271, 2304)
            if multimodal_attention_mask is not None:
                multimodal_attention_mask = rearrange(multimodal_attention_mask, "(b l) n -> b (l n)", l=seq_len)

        output = self.model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=multimodal_embeds,
            use_cache=False,
            output_hidden_states=True,
        )

        output_hs = output.hidden_states[-1].clone()
        if history_type == "pre":
            multimodal_embeds = rearrange(multimodal_embeds, "b (l n) d -> (b l) n d", l=seq_len)
            output_hs = rearrange(output_hs, "b (l n) d -> (b l) n d", l=seq_len)

        action_hs = output_hs[action_token_mask].reshape(bs, seq_len, self.latent_num, -1)  # (4, 8, 1, 1024)

        if self.use_clip_norm and mode == "train":
            clip_loss = self.clip_norm_head(action_hs, raw_text)
            self._update_loss(loss, clip_loss, "clip")

        action_logits, action_loss = self._forward_action_head(action_hs, action_labels, action_mask)

        # cur = time.time()
        # print("predict action consumes {} sec".format(cur-st))
        # st = cur

        if mode == "train":
            self._update_loss(loss, action_loss, "act")
            loss = self._format_loss(loss)
        else:
            return action_logits

        return loss

    def forward(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,  # not used
        use_cached_vision_x: bool = False,  # TODO: Do we need this? If not we can remove it from here # not used
        action_labels: Tuple[torch.Tensor, Optional[torch.Tensor]] = None,
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
        if isinstance(data_source, list):
            data_source = "_".join(data_source)
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
        action_labels: Tuple[torch.Tensor, Optional[torch.Tensor]] = None,
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
        **kwargs,
    ):

        return self.forward_continuous(
            vision_x=vision_x,
            lang_x=lang_x,
            attention_mask=attention_mask,
            action_labels=action_labels,
            action_mask=action_mask,
            vision_gripper=vision_gripper,
            raw_text=raw_text,
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
        prediction["action"] = self.forward_continuous(
            vision_x,
            lang_x,
            attention_mask,
            vision_gripper=vision_gripper,
            mode="inference",
        )

        return prediction


def deep_update(d1, d2):
    # use d2 to update d1
    for k, v in d2.items():
        if isinstance(v, dict) and k in d1:
            assert isinstance(d1[k], dict)
            deep_update(d1[k], d2[k])
        else:
            d1[k] = d2[k]
    return d1


import json


def load_config(config_file):
    _config = json.load(open(config_file))
    config = {}
    if _config.get("parent", None):
        deep_update(config, load_config(_config["parent"]))
    deep_update(config, _config)
    return config
