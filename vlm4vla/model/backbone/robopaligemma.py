import torch
from vlm4vla.model.backbone.base_backbone import BaseRoboVLM, load_config, deep_update


class RoboPaligemma(BaseRoboVLM):

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

    @property
    def model(self):
        return self.backbone

    def model_encode_images(self, images):
        image_outputs = self.model.vision_tower(images)
        selected_image_feature = image_outputs.last_hidden_state
        image_features = self.model.multi_modal_projector(selected_image_feature)
        image_features = image_features / (self.model.config.hidden_size**0.5)
        return image_features


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
