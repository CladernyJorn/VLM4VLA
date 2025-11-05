import torch
from einops import rearrange, repeat
from vlm4vla.model.backbone.base_backbone import BaseRoboVLM, load_config, deep_update
from typing import Optional, Tuple, List
from typing import Sequence


class RoboFlorence(BaseRoboVLM):

    @property
    def image_processor(self):
        # return None
        return self.processor.image_processor

    @property
    def hidden_size(self):
        return self.model.config.text_config.d_model  # 1024

    @property
    def word_embedding(self):
        return self.model.get_input_embeddings()

    @property
    def text_tower(self):
        return self.model.language_model

    @property
    def vision_tower(self):
        return self.model.vision_tower

    @property
    def model(self):
        return self.backbone

    def model_encode_images(self, images):
        image_features = self.model._encode_image(images)
        return image_features

    def forward_continuous(
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
        fwd_rgb_labels: torch.Tensor = None,
        fwd_hand_rgb_labels: torch.Tensor = None,
        fwd_mask: torch.Tensor = None,
        instr_and_action_ids=None,
        instr_and_action_labels=None,
        instr_and_action_mask=None,
        mode="train",
        **kwargs,
    ):
        loss = {}
        assert (vision_x
                is not None) or use_cached_vision_x, ("Must provide either vision_x or use_cached_vision_x to True.")
        bs, seq_len = vision_x.shape[:2]
        action_space = self.act_head_configs.get("action_space", "continuous")
        if seq_len > 1:
            lang_x = lang_x.unsqueeze(1).repeat(1, seq_len, 1).flatten(0, 1)
            attention_mask = (attention_mask.unsqueeze(1).repeat(1, seq_len, 1).flatten(0, 1))

            vision_x = vision_x.reshape(bs * seq_len, *vision_x.shape[2:]).unsqueeze(1)
            if vision_gripper is not None:
                vision_gripper = vision_gripper.reshape(bs * seq_len, *vision_gripper.shape[2:]).unsqueeze(1)

        if use_cached_vision_x:
            # Case: use cached; vision_x should be cached and other
            # vision-related inputs should not be provided.
            assert (vision_x is None), "Expect vision_x to be None when use_cached_vision_x is True."
            assert self.model.is_conditioned()

        else:
            # Case: do not use caching (i.e. this is a standard forward pass);
            self._encode_multi_vision_post_fusion(vision_x, vision_gripper)
        if action_space == "continuous":
            action_ids = torch.full((bs * seq_len, self.latent_num), self.action_token_id).to(lang_x.device)
            tmp_action_masks = torch.ones_like(action_ids)
            input_ids, action_ids_mask = self.cat_multi_input_ids(lang_x, action_ids, -1, attention_mask)
            attention_mask, _ = self.cat_multi_input_ids(attention_mask, tmp_action_masks, -1, attention_mask)
        else:
            input_ids = lang_x
        # print(lang_x.shape, attention_mask.shape)
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # past_key_values=past_key_values,
            # use_cache=use_cache,
            output_hidden_states=True,
        )

        output_hs = output.hidden_states[-1].clone()

        if self.train_setup_configs["predict_action"] and (action_labels is not None or mode == "inference"):
            # output_hs = output.hidden_states[-1].clone()
            if action_space == "continuous":
                action_hs = output_hs[action_ids_mask].reshape(bs, seq_len, self.latent_num, -1)
            elif action_space == "down_sample":
                action_hs = output_hs.reshape(bs, seq_len, *output_hs.shape[-2:])

            action_logits, action_loss = self._forward_action_head(action_hs, action_labels, action_mask)
            if mode == "train":
                self._update_loss(loss, action_loss, "act")
            else:
                return action_logits

        loss = self._format_loss(loss)

        return loss


if __name__ == "__main__":
    model_config = load_config("configs/calvin_finetune/finetune_florence_calvin.json")
    configs = model_config
    use_hand_rgb = False  # True
    model = RoboFlorence(
        configs=model_config,
        train_setup_configs=configs["train_setup"],
        fwd_head_configs=None,
        window_size=configs["window_size"],
        use_hand_rgb=use_hand_rgb,
        act_head_configs=configs["act_head"],
        fwd_pred_next_n=configs["fwd_pred_next_n"],
        use_state=True,
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Florence Model Parameters: {total_params / 1000000:.2f}M")
    # import pdb; pdb.set_trace()
    bs, seq_len = 2, 2
    device = "cuda:0"
    device = "cpu"
    dtype = torch.float32
    vision_x = torch.zeros((bs, seq_len, 3, 224, 224), dtype=dtype).to(device)
    vision_gripper = torch.zeros((bs, seq_len, 3, 224, 224), dtype=dtype).to(device)
    lang_x = torch.ones((bs, 10), dtype=torch.long).to(device) * 100
    attention_mask = torch.ones((bs, 10)).bool().to(device)
    action_lables = (
        torch.randn(bs, seq_len, configs["fwd_pred_next_n"], 6).to(device),
        torch.zeros(bs, seq_len, configs["fwd_pred_next_n"]).to(device),
    )
    model = model.to(device).to(dtype)
    rel_state = torch.randn((bs, seq_len, 7), dtype=dtype).to(device)
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
