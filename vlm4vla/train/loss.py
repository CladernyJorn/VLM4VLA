import torch
import torch.nn as nn


def calculate_vl_cross_entropy(logits, labels, mask=None):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    if mask is None:
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, logits.shape[-1]),
            shift_labels.view(-1),
        )
    else:
        # TODO: mask is with the same shape of labels,
        # 1 represents valid, 0 for non-valid, only calculate loss for valid tokens
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(
            shift_logits.view(-1, logits.shape[-1]),
            shift_labels.view(-1),
        )
        # mask the loss
        mask = mask[..., 1:].contiguous()
        loss = loss * mask.reshape(-1)
        # mean
        loss = loss.mean()
    return loss
