import torch

from vlm4vla.model.backbone.base_backbone import BaseRoboVLM, load_config, deep_update


class RoboKosMos(BaseRoboVLM):

    @property
    def hidden_size(self):
        return self.model.config.text_config.embed_dim

    @property
    def word_embedding(self):
        return self.model.text_model.model.embed_tokens

    @property
    def text_tower(self):
        return self.model.text_model.model

    @property
    def vision_tower(self):
        return self.model.vision_model

    @property
    def model(self):
        return self.backbone

    def model_encode_images(self, images):
        vision_model_output = self.model.vision_model(
            pixel_values=images,
            output_attentions=self.model.config.output_attentions,
            output_hidden_states=self.model.config.output_hidden_states,
            return_dict=self.model.config.return_dict,
        )
        # The whole `last_hidden_state` through `post_layernorm` instead of just `pooled_output`.
        image_embeds = self.model.vision_model.model.post_layernorm(vision_model_output[0])
        # normalized features
        image_embeds = torch.nn.functional.normalize(image_embeds, dim=-1)
        image_embeds, projection_attentions = self.model.image_to_text_projection(image_embeds)

        return image_embeds


if __name__ == "__main__":
    model_config = load_config(
        "configs/calvin_finetune/finetune_kosmos_cont-lstm-post_full-ft_text_vision_wd-0_ws-8_act-10.json")
    configs = model_config
    use_hand_rgb = False  # True
    model = RoboKosMos(
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
    print(f"Kosmos Model Parameters: {total_params / 1000000:.2f}M")
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
