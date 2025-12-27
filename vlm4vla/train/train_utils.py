import math
from einops import rearrange, repeat

import torch


def smooth_l1_loss(preds, targets, reduction="mean", beta=1.0):
    assert preds.size() == targets.size()
    diff = torch.abs(preds - targets)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(f"Unknown reduction mode {reduction}")
    return loss


def adjust_learning_rate(iter, configs):
    """Decay the learning rate with half-cycle cosine after warmup"""
    warmup_iters = configs["warmup_iters"]
    total_iters = configs["iters"]
    min_lr_scale = configs["min_lr_scale"]

    if iter < configs["warmup_iters"]:
        lr_scaler = 1.0 * iter / warmup_iters
    else:
        lr_scaler = min_lr_scale + (1.0 - min_lr_scale) * 0.5 * (1.0 + math.cos(math.pi * (iter - warmup_iters) /
                                                                                (total_iters - warmup_iters)))

    return lr_scaler


def claw_matrix(n, k, device="cpu"):
    upper_triangle_matrix = torch.triu(torch.ones(n, n), diagonal=0).to(device)
    lower_triangle_matrix = torch.tril(torch.ones(n, n), diagonal=k).to(device)

    claw = upper_triangle_matrix * lower_triangle_matrix

    return claw


def generate_chunck_data(data, window_size, chunk_size):
    bs, seq_len = data.shape[:2]
    raw_data_shape = data.shape[2:]
    data_flatten = data.flatten().view(bs, seq_len, -1)
    assert (seq_len == window_size + chunk_size), f"The sequence length should be {window_size + chunk_size}"
    data_flatten = repeat(data_flatten, "b s d -> b w s d", w=window_size)

    mask = claw_matrix(seq_len, chunk_size, data_flatten.device)
    mask = mask - torch.diag_embed(mask.diag())  # set current obs mask to 0
    mask = mask[:window_size].bool()

    mask = repeat(mask, "w s -> b w s d", b=bs, d=data_flatten.shape[-1])
    data_flatten = torch.masked_select(data_flatten, mask)

    data_flatten = data_flatten.view(bs, window_size, chunk_size, *raw_data_shape)

    return data_flatten


def preprocess_text_flamingo(sample, tokenizer):
    tokenizer.padding_side = "right"
    sample = [(f"<image>{s.strip()}<|endofchunk|>{tokenizer.eos_token}") for s in sample]
    text = tokenizer(
        sample,
        max_length=32,
        padding="longest",
        truncation="only_first",
        return_tensors="pt",
    )
    return text["input_ids"], text["attention_mask"]


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


def get_checkpoint(model):
    state_dict = model.state_dict()

    for name, p in model.named_parameters():
        if not p.requires_grad and "normalizer" not in name:
            del state_dict[name]

    return state_dict


if __name__ == "__main__":
    window_size = 5
    chunck_size = 3
    bs = 2
    obs = torch.randn(bs, window_size + chunck_size, 3, 224, 224)

    future_obs_target = generate_chunck_data(obs, window_size, chunck_size)
    print(future_obs_target.shape)
