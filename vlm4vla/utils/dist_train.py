import io
import os

import torch
import torch.distributed as dist


_print = print


def get_world_size():
    return int(os.getenv("WORLD_SIZE", 1))


def get_rank():
    return int(os.getenv("RANK", 0))


def get_local_rank():
    return int(os.getenv("LOCAL_RANK", 0))


def is_dist():
    return dist.is_available() and dist.is_initialized() and get_world_size() > 1


def print(*argc, all=False, **kwargs):
    if not is_dist():
        _print(*argc, **kwargs)
        return

    if not all and get_local_rank() != 0:
        return

    output = io.StringIO()
    kwargs["end"] = ""
    kwargs["file"] = output
    kwargs["flush"] = True
    _print(*argc, **kwargs)

    s = output.getvalue()
    output.close()

    s = "[rank {}] {}".format(dist.get_rank(), s)
    _print(s)


def reduce_mean(tensor, nprocs=None):
    if not is_dist():
        return tensor
    if not isinstance(tensor, torch.Tensor):
        device = torch.cuda.current_device()
        rt = torch.tensor(tensor, device=device)
    else:
        rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    nprocs = nprocs if nprocs else dist.get_world_size()
    rt = rt / nprocs
    if not isinstance(tensor, torch.Tensor):
        rt = rt.item()
    return rt


def reduce_sum(tensor):
    if not is_dist():
        return tensor
    if not isinstance(tensor, torch.Tensor):
        device = torch.cuda.current_device()
        rt = torch.tensor(tensor, device=device)
    else:
        rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if not isinstance(tensor, torch.Tensor):
        rt = rt.item()
    return rt
