import os
import sys
from typing import Optional, Tuple, List

import torch
import torch.nn as nn

from .base_policy import BasePolicyHead

from .utils.fm_head_utils import DiT, ActionEncoder


class MLP(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(torch.relu(self.layer1(x)))


class FMDecoder(BasePolicyHead):
    """
    Flow-Matching action decoder adopting diffusion-based DiT with layer-wise cross-attention.
    Implementation mirrors starVLA's LayerwiseFlowmatchingActionHead. Dimensions follow VLM4VLA settings.
    """

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        action_dim: int,
        down_sample: str,
        latent: int,
        fwd_pred_next_n: int,
        window_size: int,
        DiTConfig: dict,
        add_pos_embed: bool = True,
        num_inference_timesteps: int = 20,
        num_timestep_buckets: int = 1000,
        noise_beta_alpha: float = 1.5,
        noise_beta_beta: float = 1.0,
        noise_s: float = 0.999,
        state_dim: int = 0,
        num_target_vision_tokens: int = 32,
        max_seq_len: int = 1024,
        **kwargs,
    ):
        super().__init__(hidden_size, action_dim, **kwargs)

        # Cache core dims
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.latent = latent
        self.fwd_pred_next_n = fwd_pred_next_n
        self.window_size = window_size

        # Build DiT exactly as starVLA: input_embedding_dim comes from DiTConfig
        if isinstance(DiTConfig, dict) and "input_embedding_dim" in DiTConfig:
            self.input_embedding_dim = DiTConfig["input_embedding_dim"]
        else:
            # Fallback to in_features if not in DiTConfig
            self.input_embedding_dim = in_features

        # Build DiT config: start with input_embedding_dim, merge with DiTConfig
        action_model_cfg = {"input_embedding_dim": self.input_embedding_dim}
        diffusion_model_cfg = {**action_model_cfg, **DiTConfig} if isinstance(DiTConfig, dict) else action_model_cfg
        self.model = DiT(**diffusion_model_cfg)

        # Encoders/decoders identical to starVLA - use input_embedding_dim consistently
        self.state_encoder = (
            MLP(input_dim=state_dim, hidden_dim=self.hidden_size, output_dim=self.input_embedding_dim)
            if state_dim else None)
        self.action_encoder = ActionEncoder(action_dim=action_dim, hidden_size=self.input_embedding_dim)
        self.action_decoder = MLP(input_dim=self.hidden_size, hidden_dim=self.hidden_size, output_dim=action_dim)
        self.future_tokens = nn.Embedding(num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        self.add_pos_embed = add_pos_embed
        if self.add_pos_embed:
            self.position_embedding = nn.Embedding(max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        from torch.distributions import Beta

        self.beta_dist = Beta(noise_beta_alpha, noise_beta_beta)
        self.noise_s = noise_s
        self.num_timestep_buckets = num_timestep_buckets
        self.num_inference_timesteps = num_inference_timesteps

        # Buffers for loss computation (set during forward)
        self._last_velocity = None

    def sample_time(self, batch_size: int, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device=device, dtype=dtype)
        # return (self.noise_s - sample) / self.noise_s
        return self.noise_s*(1-sample)

    def _ensure_vl_list(self, vl_embs_list: List[torch.Tensor], expected_len: int):
        assert isinstance(vl_embs_list, list) and len(vl_embs_list) >= expected_len, (
            "vl_embs_list must be a list of per-layer embeddings with length >= DiT layers")

    def _reshape_vl_embs_for_crossattn(self, vl_embs: torch.Tensor, bs: int, seq_len: int):
        """
        Reshape vl_embs from backbone output to (B, seq_len, hidden_dim) format for cross-attention.
        Backbone outputs may be (bs*seq_len, seq_length, hidden_dim) or (bs, seq_len*seq_length, hidden_dim).
        We need to reshape to (B, seq_length, hidden_dim) where B = bs * seq_len for cross-attention.
        """
        if vl_embs.ndim == 3:
            B_total, seq_length, hidden_dim = vl_embs.shape
            if B_total == bs * seq_len:
                # Already in the right format: (bs*seq_len, seq_length, hidden_dim)
                return vl_embs
            elif B_total == bs:
                # Need to reshape from (bs, seq_len*seq_length, hidden_dim) to (bs*seq_len, seq_length, hidden_dim)
                # Assuming seq_length represents tokens per timestep
                if seq_length % seq_len == 0:
                    tokens_per_step = seq_length // seq_len
                    vl_embs = vl_embs.reshape(bs, seq_len, tokens_per_step, hidden_dim)
                    vl_embs = vl_embs.reshape(bs * seq_len, tokens_per_step, hidden_dim)
                    return vl_embs
                else:
                    # Fallback: treat as single timestep
                    return vl_embs.repeat_interleave(seq_len, dim=0)
            else:
                # Try to infer the correct reshape
                return vl_embs.reshape(bs * seq_len, -1, hidden_dim)
        return vl_embs

    def _concat_state_future_action(self, B: int, action_features: torch.Tensor,
                                    state_features: Optional[torch.Tensor]):
        device = action_features.device
        # Add pos embedding
        if self.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(B, -1, -1)
        if state_features is not None:
            sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)
        else:
            sa_embs = torch.cat((future_tokens, action_features), dim=1)
        return sa_embs

    def forward(
        self,
        tok_seq: torch.Tensor,
        actions: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        action_masks: Optional[torch.Tensor] = None,
        vl_embs_list: Optional[List[torch.Tensor]] = None,
        state: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # tok_seq shape: (B, L, latent, D) or (B, L, D) — we don't use it in FM; rely on vl_embs_list instead
        assert vl_embs_list is not None, "FMDecoder requires vl_embs_list from backbone hidden states"

        # Unpack labels into a continuous action vector: (..., 6)+( ..., 1) -> (..., 7)
        if actions is None or actions[0] is None:
            raise ValueError("FMDecoder forward requires ground-truth actions during training for flow matching.")

        arm, gripper = actions  # shapes: (bs, seq, T, 6), (bs, seq, T)
        if gripper.ndim == arm.ndim:
            gripper = gripper[..., -1]
        full_actions = torch.cat([arm, gripper.unsqueeze(-1)], dim=-1)

        # Flatten batch and sequence dims to match how backbone feeds the LLM (bs*seq)
        bs, seq_len, T, _ = full_actions.shape
        B = bs * seq_len
        actions_bt = full_actions.reshape(B, T, self.action_dim)

        # Prepare state features if provided (bs, seq, state_dim) -> (B, 1, in_features)
        state_features = None
        if state is not None and self.state_encoder is not None:
            state_bt = state.reshape(B, state.shape[-1])
            state_features = self.state_encoder(state_bt).unsqueeze(1)

        # Reshape vl_embs_list for cross-attention: each should be (B, seq_length, hidden_dim)
        expected_layers = len(self.model.transformer_blocks)
        self._ensure_vl_list(vl_embs_list, expected_len=expected_layers)

        # Take the last N layers to match DiT depth and reshape
        vl_embs_list_reshaped = []
        for vl_embs in vl_embs_list[-expected_layers:]:
            vl_embs_reshaped = self._reshape_vl_embs_for_crossattn(vl_embs, bs, seq_len)
            vl_embs_list_reshaped.append(vl_embs_reshaped)

        device = actions_bt.device
        dtype = actions_bt.dtype

        # Sample noise and time t, construct noisy trajectory and velocity (as in starVLA)
        noise = torch.randn(actions_bt.shape, device=device, dtype=dtype)
        t_cont = self.sample_time(B, device=device, dtype=dtype)  # (B,)
        t = t_cont[:, None, None]
        noisy_trajectory = (1 - t) * noise + t * actions_bt
        velocity = actions_bt - noise

        # Discretize t for embedding (B,)
        t_discretized = (t_cont * self.num_timestep_buckets).long()

        # Encode actions with timestep
        action_features = self.action_encoder(noisy_trajectory, t_discretized)

        # Encode state+future+action into SA sequence
        sa_embs = self._concat_state_future_action(B, action_features, state_features)

        # Timestep embedding once
        temb = self.model.timestep_encoder(t_discretized)

        # Layerwise cross-attention
        model_output = sa_embs
        for layer_idx, layer in enumerate(self.model.transformer_blocks):
            model_output = layer(
                hidden_states=model_output,
                encoder_hidden_states=vl_embs_list_reshaped[layer_idx],
                temb=temb,
            )

        # Apply DiT's output projection (norm_out + proj_out_2) to reduce dimension from inner_dim to output_dim
        # This matches starVLA's DiT forward method
        conditioning = temb
        shift, scale = self.model.proj_out_1(torch.nn.functional.silu(conditioning)).chunk(2, dim=1)
        model_output = self.model.norm_out(model_output) * (1 + scale[:, None]) + shift[:, None]
        model_output = self.model.proj_out_2(model_output)

        pred = self.action_decoder(model_output)
        pred_actions = pred[:, -T:]

        # Cache velocity for loss()
        self._last_velocity = velocity.detach()

        # Restore to (bs, seq, T, action_dim)
        pred_actions = pred_actions.reshape(bs, seq_len, T, self.action_dim)
        return pred_actions

    def get_labels(self, pred_actions, labels, action_masks, **kwargs):
        # Return labels as continuous action vector to match pred_actions
        if labels is None or labels[0] is None:
            return pred_actions, labels, action_masks
        arm, gripper = labels
        if gripper.ndim == arm.ndim:
            gripper = gripper[..., -1]
        full_actions = torch.cat([arm, gripper.unsqueeze(-1)], dim=-1)
        return pred_actions, full_actions, action_masks

    def loss(self, pred_action, labels, attention_mask=None):
        if labels is None:
            return {"loss": None}
        # pred_action: (bs, seq, T, D), labels: (bs, seq, T, D)
        B, S, T, D = pred_action.shape
        pred_bt = pred_action.reshape(B * S, T, D)
        if self._last_velocity is None:
            raise RuntimeError("FMDecoder.loss called before forward or cache cleared.")
        target_bt = self._last_velocity
        if attention_mask is not None:
            mask = attention_mask.reshape(B * S, T).bool()
            loss = ((pred_bt - target_bt)**2)[mask].mean()
        else:
            loss = ((pred_bt - target_bt)**2).mean()
        return {"loss_arm": loss, "loss_gripper": torch.tensor(0.0, device=pred_action.device), "acc_gripper": -1.0}

    @torch.no_grad()
    def predict(self, vl_embs_list: List[torch.Tensor], state: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Sampling loop (Euler) as in starVLA
        B = vl_embs_list[0].shape[0]
        device = vl_embs_list[0].device
        actions = torch.randn(
            size=(B, self.fwd_pred_next_n, self.action_dim), dtype=vl_embs_list[0].dtype, device=device)
        dt = 1.0 / float(self.num_inference_timesteps)
        state_features = None
        if state is not None and self.state_encoder is not None:
            state_features = self.state_encoder(state).unsqueeze(1)
        for t_step in range(self.num_inference_timesteps):
            t_cont = t_step / float(self.num_inference_timesteps)
            t_disc = int(t_cont * self.num_timestep_buckets)
            timesteps_tensor = torch.full(size=(B,), fill_value=t_disc, device=device, dtype=torch.long)
            action_features = self.action_encoder(actions, timesteps_tensor)
            sa_embs = self._concat_state_future_action(B, action_features, state_features)
            temb = self.model.timestep_encoder(timesteps_tensor)
            model_output = sa_embs

            # Reshape vl_embs_list for cross-attention if needed
            expected_layers = len(self.model.transformer_blocks)
            vl_embs_list_reshaped = []
            for vl_embs in vl_embs_list[-expected_layers:]:
                # For inference, B is already the flattened batch size
                if vl_embs.ndim == 3 and vl_embs.shape[0] != B:
                    # Need to reshape
                    vl_embs = vl_embs.reshape(B, -1, vl_embs.shape[-1])
                vl_embs_list_reshaped.append(vl_embs)

            for layer_idx, layer in enumerate(self.model.transformer_blocks):
                model_output = layer(
                    hidden_states=model_output,
                    encoder_hidden_states=vl_embs_list_reshaped[layer_idx],
                    temb=temb,
                )

            # Apply DiT's output projection (norm_out + proj_out_2) to reduce dimension from inner_dim to output_dim
            conditioning = temb
            shift, scale = self.model.proj_out_1(torch.nn.functional.silu(conditioning)).chunk(2, dim=1)
            model_output = self.model.norm_out(model_output) * (1 + scale[:, None]) + shift[:, None]
            model_output = self.model.proj_out_2(model_output)

            pred = self.action_decoder(model_output)
            pred_velocity = pred[:, -self.fwd_pred_next_n:]
            actions = actions + dt * pred_velocity
        return actions[...,:6],actions[...,-1]  # Return (arm, gripper)


if __name__ == "__main__":
    # Test FMDecoder with dummy data
    print("Testing FMDecoder...")

    # Test parameters
    batch_size = 2
    seq_len = 8
    action_dim = 7
    T = 10  # action horizon
    in_features = 2560
    hidden_size = 1024

    # Create DiTConfig
    DiTConfig = {
        "input_embedding_dim": 1536,
        "num_attention_heads": 16,
        "attention_head_dim": 96,
        "num_layers": 12,
        "cross_attention_dim": hidden_size,  # VL embedding dimension
    }

    # Create FMDecoder
    fm_decoder = FMDecoder(
        in_features=in_features,
        hidden_size=hidden_size,
        action_dim=action_dim,
        down_sample="none",
        latent=1,
        fwd_pred_next_n=T,
        window_size=seq_len,
        DiTConfig=DiTConfig,
        num_inference_timesteps=20,
    )

    # Create dummy inputs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fm_decoder = fm_decoder.to(device)

    # Create dummy tok_seq (not used but required by interface)
    tok_seq = torch.randn(batch_size, seq_len, 1, hidden_size).to(device)

    # Create dummy actions: (arm, gripper)
    arm = torch.randn(batch_size, seq_len, T, 6).to(device)
    gripper = torch.randn(batch_size, seq_len, T).to(device)
    actions = (arm, gripper)

    # Create dummy vl_embs_list: list of hidden states from backbone
    # Each should be (batch_size * seq_len, seq_length, hidden_dim)
    # For test, we'll use (batch_size, seq_len * seq_length, hidden_dim) and let it reshape
    num_layers = DiTConfig["num_layers"]
    vl_embs_list = []
    for _ in range(num_layers + 5):  # More layers than needed, will take last N
        # Simulate backbone output: (batch_size, seq_len * tokens_per_step, hidden_dim)
        # In real case, this comes from backbone hidden_states
        vl_embs = torch.randn(batch_size, seq_len * 256, hidden_size).to(device)
        vl_embs_list.append(vl_embs)

    print(f"Created FMDecoder with {sum(p.numel() for p in fm_decoder.parameters()) / 1e6:.2f}M parameters")
    print(f"Testing forward pass...")

    # Test forward pass
    try:
        pred_actions = fm_decoder(
            tok_seq=tok_seq,
            actions=actions,
            vl_embs_list=vl_embs_list,
        )
        print(f"Forward pass successful! Output shape: {pred_actions.shape}")
        assert pred_actions.shape == (
            batch_size, seq_len, T,
            action_dim), f"Expected {(batch_size, seq_len, T, action_dim)}, got {pred_actions.shape}"

        # Test loss
        labels = (arm, gripper)
        loss_dict = fm_decoder.loss(pred_actions, labels)
        print(f"Loss computation successful! Loss: {loss_dict}")

        # Test inference (predict)
        print("Testing inference (predict)...")
        # For inference, vl_embs_list should be reshaped to (B, seq_length, hidden_dim)
        # where B = batch_size * seq_len (or just batch_size for single-step inference)
        vl_embs_list_inference = []
        for vl_embs in vl_embs_list[-num_layers:]:
            # Reshape to (batch_size * seq_len, tokens_per_step, hidden_dim)
            B_total = batch_size * seq_len
            tokens_per_step = 256
            vl_embs_inf = vl_embs.reshape(B_total, tokens_per_step, hidden_size)
            vl_embs_list_inference.append(vl_embs_inf)

        with torch.no_grad():
            pred_actions_inf = fm_decoder.predict(vl_embs_list_inference)
        print(f"Inference successful! Output shape: {pred_actions_inf.shape}")
        assert pred_actions_inf.shape == (
            B_total, T, action_dim), f"Expected {(B_total, T, action_dim)}, got {pred_actions_inf.shape}"

        print("\n✓ All tests passed!")

    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
