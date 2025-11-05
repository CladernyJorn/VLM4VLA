import torch
import functools
import logging
import time


def rotate_half(x):
    # Build the [-x2, x1, -x4, x3, ...] tensor for the sin part of the positional encoding.
    x1 = x[..., :x.shape[-1] // 2]  # Takes the first half of the last dimension
    x2 = x[..., x.shape[-1] // 2:]  # Takes the second half of the last dimension
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)  # Add the head dimension
    sin = sin.unsqueeze(unsqueeze_dim)  # Add the head dimension
    # Apply the formula (34) of the Rotary Positional Encoding paper.
    x = (x * cos) + (rotate_half(x) * sin)
    return x


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(
        batch,
        num_key_value_heads * n_rep,
        slen,
        head_dim,
    )


def conditional_decorator(dec, condition):

    def decorator(func):
        if not condition:
            # Return the function unchanged, not decorated.
            return func
        return dec(func)

    return decorator


class NoSyncBase:

    def no_sync(self):
        if self.use_ddp:
            # If DDP is used, call the actual `no_sync` method
            return torch.nn.parallel.DistributedDataParallel.no_sync(self)
        else:
            # Otherwise, return the dummy context manager
            class DummyContext:

                def __enter__(self):
                    pass

                def __exit__(self, exc_type, exc_value, traceback):
                    pass

            return DummyContext()


def main_rank_only(func):

    def wrapper(*args, **kwargs):
        if not kwargs.get("main_rank", False):
            return None
        return func(*args, **kwargs)

    return wrapper


def log_allocated_gpu_memory(log=None, stage="loading model", device=0):
    if torch.cuda.is_available():
        allocated_memory = torch.cuda.memory_allocated(device)
        msg = f"Allocated GPU memory after {stage}: {allocated_memory/1024/1024/1024:.2f} GB"
        print(msg) if log is None else log.info(msg)


def log_execution_time(logger=None):
    """Decorator to log the execution time of a function"""

    def decorator(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            if logger is None:
                print(f"{func.__name__} took {elapsed_time:.2f} seconds to execute.")
            else:
                logger.info(f"{func.__name__} took {elapsed_time:.2f} seconds to execute.")
            return result

        return wrapper

    return decorator


class Timer:

    def __init__(self):
        self._start = time.time()

    def __call__(self, reset=True):
        now = time.time()
        diff = now - self._start
        if reset:
            self._start = now
        return diff


# Filter to log only on the main rank
class MainRankFilter(logging.Filter):

    def __init__(self, main_rank):
        super().__init__()
        self.main_rank = main_rank

    def filter(self, record):
        # Only log if this is the main rank
        return self.main_rank
