from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPTanhHead(torch.nn.Module):

    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, output_size),
            torch.nn.Tanh(),
        )

    def forward(self, x):
        return self.mlp(x)


class MLPNohHead(torch.nn.Module):

    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, output_size),
        )

    def forward(self, x):
        return self.mlp(x)


class MLPSigmoidHead(torch.nn.Module):

    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, output_size),
            # torch.nn.Sigmoid(),
        )

    def forward(self, x):
        return self.mlp(x)


class MLPHead(torch.nn.Module):

    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, output_size),
        )

    def forward(self, x):
        return self.mlp(x)


class BasePolicyHead(nn.Module):

    def __init__(
        self,
        hidden_size,
        action_dim,
        action_space="continuous",
        down_sample="pooling",
        latent=1,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim

        self.down_sample = down_sample
        self.latent = latent
        self.action_space = action_space

    @staticmethod
    def _get_target_modal_tokens(tok_seq, tok_mask):
        index = tok_mask.nonzero(as_tuple=True)
        return tok_seq[index]

    def get_modal_tokens(self, tok_seq, tok_mask_dict, modal_name):
        assert modal_name in tok_mask_dict, f"{modal_name} not in token sequence"
        return self._get_target_modal_tokens(tok_seq, tok_mask_dict[modal_name])

    def loss(self, pred_action, labels, attention_mask=None):
        """
        pred_action_logits: [bs, seq_len, chunck_size, 7], 1-6 refers to ee pose, 7 refers to gripper open/close
        lables: (pose gt [bs, seq_len, chunck_size, 6], gripper gt [bs, seq_len, chunck_size])
        attention_mask: [bs, seq_len, chunck_size]
        """
        if labels is None or labels[0] is None:
            return {"loss": None}

        if isinstance(pred_action, tuple) or isinstance(pred_action, list):
            if pred_action[0].ndim == pred_action[1].ndim:
                pred_action = torch.cat(pred_action, dim=-1)
            elif pred_action[0].ndim == pred_action[1].ndim + 1:
                pred_action = torch.cat([pred_action[0], pred_action[1].unsqueeze(-1)], dim=-1)
            else:
                raise ValueError("Can not solve the gripper action dim")
        if attention_mask is None:
            pose_loss = torch.nn.functional.huber_loss(pred_action[..., :6], labels[0])
            # pose_loss = torch.nn.functional.mse_loss(pred_action[..., :6], labels[0])
            gripper_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_action[..., -1], labels[1])
        else:
            pose_loss = torch.nn.functional.huber_loss(pred_action[..., :6], labels[0], reduction="none")
            # pose_loss = torch.nn.functional.mse_loss(pred_action[..., :6], labels[0], reduction='none')
            attention_mask = attention_mask.bool()
            pose_loss = pose_loss[attention_mask].mean()
            # gripper_loss = torch.nn.functional.binary_cross_entropy(pred_action[..., -1], labels[1], reduction='none')
            gripper_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                pred_action[..., -1], labels[1], reduction="none")
            gripper_loss = gripper_loss[attention_mask].mean()

        gripper_action_preds = (F.sigmoid(pred_action[..., -1]) > 0.5).float()
        acc_gripper_act = torch.eq(gripper_action_preds, labels[1]).float()
        if attention_mask is None:
            acc_gripper_act = acc_gripper_act.mean()
        else:
            # acc_gripper_act = (acc_gripper_act * attention_mask).sum() / attention_mask.sum()
            acc_gripper_act = acc_gripper_act[attention_mask].mean()

        return {
            "loss_arm": pose_loss,
            "loss_gripper": gripper_loss,
            "acc_gripper": acc_gripper_act.item(),
        }

    def get_labels(self, pred_actions, labels, action_masks, **kwargs):
        return pred_actions, labels, action_masks


def initialize_param(model):
    with torch.no_grad():
        for m in model.children():
            if hasattr(m, "weight"):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, "bias"):
                    m.bias.fill_(0)
            else:
                initialize_param(m)


class FCDecoder(BasePolicyHead):

    def __init__(
        self,
        in_features,
        hidden_size,
        action_dim,
        down_sample,
        latent,
        fwd_pred_next_n,
        **kwargs,
    ):
        super(FCDecoder, self).__init__(hidden_size, action_dim, **kwargs)
        self.down_sample = down_sample
        self.latent = latent
        self.fwd_pred_next_n = fwd_pred_next_n
        self.actions = MLPTanhHead(self.hidden_size * latent, fwd_pred_next_n * (self.action_dim - 1))
        self.gripper = MLPSigmoidHead(self.hidden_size * latent, fwd_pred_next_n)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features * latent, in_features * latent // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features * latent // 2, hidden_size * latent),
        )
        if self.down_sample == "pooling":
            self.global_1d_pool = nn.AdaptiveMaxPool1d(latent)
        elif self.down_sample == "resampler":
            pass
        elif self.down_sample == "none":
            pass
        else:
            raise NotImplementedError
        initialize_param(self)

    def forward(self, tok_seq, **kwargs):
        if len(tok_seq.shape) == 4:
            bs, seq_len, n_tok, tok_dim = tok_seq.shape
            tok_seq = rearrange(tok_seq,
                                "b l n d-> (b l) n d")  # reduce the seq_len dim (4, 8, 1, 1024)->(4*8, 1, 1024)
        elif tok_seq.dim() == 3:
            bs, n_tok, tok_dim = tok_seq.shape
            seq_len = None
        else:
            assert len(tok_seq.shape) == 2
            bs, tok_dim = tok_seq.shape
            seq_len = None
            n_tok = None
            tok_seq = tok_seq.unsqueeze(1)

        # here tok_seq is (bs*seq_len, n_tok, tok_dim)
        if self.down_sample == "pooling":
            tok_seq = self.global_1d_pool(tok_seq.permute(0, 2, 1))
            tok_seq = rearrange(tok_seq, "b d n -> b (n d)")
        elif self.down_sample == "resampler":
            raise NotImplementedError
        elif self.down_sample == "none":
            tok_seq = rearrange(tok_seq, "b n d -> b (n d)")
        else:
            raise NotImplementedError

        tok_seq = self.mlp(tok_seq)
        actions = self.actions(tok_seq)
        gripper = self.gripper(tok_seq)
        if seq_len is not None:
            # input is 4-dim
            actions = rearrange(
                actions,
                "(b l) (n d) -> b l n d",
                b=bs,
                l=seq_len,
                n=self.fwd_pred_next_n,
            )
            gripper = rearrange(
                gripper,
                "(b l) (n d) -> b l n d",
                b=bs,
                l=seq_len,
                n=self.fwd_pred_next_n,
            )
        elif n_tok is not None:
            # input is 3-dim
            actions = rearrange(actions, "b (n d) -> b n d", b=bs, n=self.fwd_pred_next_n)
            gripper = rearrange(gripper, "b (n d) -> b n d", b=bs, n=self.fwd_pred_next_n)

        return actions, gripper


class FCDecoder_dualarm(BasePolicyHead):

    def __init__(
        self,
        in_features,
        hidden_size,
        action_dim,
        down_sample,
        latent,
        fwd_pred_next_n,
        **kwargs,
    ):
        super(FCDecoder_dualarm, self).__init__(hidden_size, action_dim, **kwargs)
        self.down_sample = down_sample
        self.latent = latent
        self.fwd_pred_next_n = fwd_pred_next_n
        assert self.action_dim == 14, "action_dim must be 14 (left 7 + right 7)"
        self.actions = MLPTanhHead(self.hidden_size * latent, fwd_pred_next_n * (self.action_dim - 2))
        self.gripper = MLPSigmoidHead(self.hidden_size * latent, fwd_pred_next_n * 2)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features * latent, in_features * latent // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features * latent // 2, hidden_size * latent),
        )
        if self.down_sample == "pooling":
            self.global_1d_pool = nn.AdaptiveMaxPool1d(latent)
        elif self.down_sample == "resampler":
            pass
        elif self.down_sample == "none":
            pass
        else:
            raise NotImplementedError
        initialize_param(self)

    def forward(self, tok_seq, **kwargs):
        if len(tok_seq.shape) == 4:
            bs, seq_len, n_tok, tok_dim = tok_seq.shape
            tok_seq = rearrange(tok_seq,
                                "b l n d-> (b l) n d")  # reduce the seq_len dim (4, 8, 1, 1024)->(4*8, 1, 1024)
        elif tok_seq.dim() == 3:
            bs, n_tok, tok_dim = tok_seq.shape
            seq_len = None
        else:
            assert len(tok_seq.shape) == 2
            bs, tok_dim = tok_seq.shape
            seq_len = None
            n_tok = None
            tok_seq = tok_seq.unsqueeze(1)

        # here tok_seq is (bs*seq_len, n_tok, tok_dim)
        if self.down_sample == "pooling":
            tok_seq = self.global_1d_pool(tok_seq.permute(0, 2, 1))
            tok_seq = rearrange(tok_seq, "b d n -> b (n d)")
        elif self.down_sample == "resampler":
            raise NotImplementedError
        elif self.down_sample == "none":
            tok_seq = rearrange(tok_seq, "b n d -> b (n d)")
        else:
            raise NotImplementedError

        tok_seq = self.mlp(tok_seq)
        actions = self.actions(tok_seq)
        gripper = self.gripper(tok_seq)
        if seq_len is not None:
            # input is 4-dim
            actions = rearrange(
                actions,
                "(b l) (n x d) -> b l n x d",
                b=bs,
                l=seq_len,
                n=self.fwd_pred_next_n,
                x=2,
            )
            gripper = rearrange(
                gripper,
                "(b l) (n x d) -> b l n x d",
                b=bs,
                l=seq_len,
                n=self.fwd_pred_next_n,
                x=2,
            )
        elif n_tok is not None:
            # input is 3-dim
            actions = rearrange(actions, "b (n x d) -> b n x d", b=bs, n=self.fwd_pred_next_n, x=2)
            gripper = rearrange(gripper, "b (n x d) -> b n x d", b=bs, n=self.fwd_pred_next_n, x=2)

        return actions, gripper
