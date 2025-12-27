"""
traj_transforms.py

Contains trajectory transforms used in the orca data pipeline. Trajectory transforms operate on a dictionary
that represents a single trajectory, meaning each tensor has the same leading dimension (the trajectory length).
"""

from asyncio import gather
import logging
from typing import Dict, Literal

import tensorflow as tf


def chunk_act_obs(traj: Dict, window_size: int, future_action_window_size: int = 0) -> Dict:
    """
    Chunks actions and observations into the given window_size.

    "observation" keys are given a new axis (at index 1) of size `window_size` containing `window_size - 1`
    observations from the past and the current observation. "action" is given a new axis (at index 1) of size
    `window_size + future_action_window_size` containing `window_size - 1` actions from the past, the current
    action, and `future_action_window_size` actions from the future. "pad_mask" is added to "observation" and
    indicates whether an observation should be considered padding (i.e. if it had come from a timestep
    before the start of the trajectory).
    """
    traj_len = tf.shape(traj["action"])[0]
    action_dim = traj["action"].shape[-1]

    action_chunk_indices = tf.broadcast_to(
        tf.range(-window_size + 1, 1 + future_action_window_size),
        [traj_len, window_size + future_action_window_size],
    ) + tf.broadcast_to(
        tf.range(traj_len)[:, None],
        [traj_len, window_size + future_action_window_size],
    )

    if "timestep" in traj["task"]:
        goal_timestep = traj["task"]["timestep"]
    else:
        goal_timestep = tf.fill([traj_len], traj_len - 1)

    floored_action_chunk_indices = tf.minimum(tf.maximum(action_chunk_indices, 0), goal_timestep[:, None])

    traj["chunk_mask"] = (action_chunk_indices >= 0) & (action_chunk_indices <= goal_timestep[:, None])
    traj["observation"] = tf.nest.map_structure(lambda x: tf.gather(x, floored_action_chunk_indices), traj["observation"])
    traj["action"] = tf.gather(traj["action"], floored_action_chunk_indices)

    # if no absolute_action_mask was provided, assume all actions are relative
    if "absolute_action_mask" not in traj and future_action_window_size > 0:
        logging.warning(
            "future_action_window_size > 0 but no absolute_action_mask was provided. "
            "Assuming all actions are relative for the purpose of making neutral actions."
        )
    absolute_action_mask = traj.get("absolute_action_mask", tf.zeros([traj_len, action_dim], dtype=tf.bool))
    neutral_actions = tf.where(
        absolute_action_mask[:, None, :],
        traj["action"],  # absolute actions are repeated (already done during chunking)
        tf.zeros_like(traj["action"]),  # relative actions are zeroed
    )

    # actions past the goal timestep become neutral
    action_past_goal = action_chunk_indices > goal_timestep[:, None]
    traj["action"] = tf.where(action_past_goal[:, :, None], neutral_actions, traj["action"])

    return traj


def new_chunk_act_obs(traj: Dict, window_size: int, future_action_window_size: int = 0, left_pad: bool=True, window_sample: Literal["sliding", "range"]="sliding") -> Dict:
    """
    Chunks actions and observations into the given window_size.

    "observation" keys are given a new axis (at index 1) of size `window_size` containing `window_size - 1`
    observations from the past and the current observation. "action" is given a new axis (at index 1) of size
    `window_size + future_action_window_size` containing `window_size - 1` actions from the past, the current
    action, and `future_action_window_size` actions from the future. "pad_mask" is added to "observation" and
    indicates whether an observation should be considered padding (i.e. if it had come from a timestep
    before the start of the trajectory).
    """
    traj_len = tf.shape(traj["action"])[0]
    traj = chunk_act_obs(traj, window_size, future_action_window_size)
    left_index = 0 if left_pad else window_size - 1
    tf.assert_less(left_index, traj_len)
    def slice_first_dim(x):
        return x[left_index:]
    
    def repeat_first_dim(x):
        return tf.repeat(x, repeats=window_size, axis=0)

    traj = tf.nest.map_structure(slice_first_dim, traj)
    if window_sample == "range":
        traj = tf.nest.map_structure(repeat_first_dim, traj)
        left_range = tf.range(window_size)
        left_range_mask = ~tf.sequence_mask(left_range, window_size + future_action_window_size)
        left_range_mask = tf.tile(left_range_mask, [traj_len-left_index, 1])
        traj["chunk_mask"] = traj["chunk_mask"] & left_range_mask
        
    return traj


def chunk_as_episode(traj: Dict, frame_num: int) -> Dict:
    traj_len = tf.shape(traj['action'])[0]
    indices = tf.cast(tf.linspace(0.0, tf.cast(traj_len - 1, tf.float32), frame_num), tf.int32)
    except_keys = ['action', 'observation']
    
    def gather_element(data):
        if isinstance(data, dict):
            for key in data:
                data[key] = gather_element(data[key])
            return data
        else:
            sampled_tensor = tf.gather(data, indices, axis=0)
            return sampled_tensor

    def get_first_element(data):
        if isinstance(data, dict):
            for key in data:
                if key in except_keys:
                    continue
                data[key] = get_first_element(data[key])
            return data
        else:
            return data[-1]

    for key in except_keys:
        traj[key] = gather_element(traj[key])
    
    return get_first_element(traj)


def subsample(traj: Dict, subsample_length: int) -> Dict:
    """Subsamples trajectories to the given length."""
    traj_len = tf.shape(traj["action"])[0]
    if traj_len > subsample_length:
        indices = tf.random.shuffle(tf.range(traj_len))[:subsample_length]
        traj = tf.nest.map_structure(lambda x: tf.gather(x, indices), traj)

    return traj


def add_pad_mask_dict(traj: Dict) -> Dict:
    """
    Adds a dictionary indicating which elements of the observation/task should be treated as padding.
        =>> traj["observation"|"task"]["pad_mask_dict"] = {k: traj["observation"|"task"][k] is not padding}
    """
    traj_len = tf.shape(traj["action"])[0]

    for key in ["observation", "task"]:
        pad_mask_dict = {}
        for subkey in traj[key]:
            # Handles "language_instruction", "image_*", and "depth_*"
            if traj[key][subkey].dtype == tf.string:
                pad_mask_dict[subkey] = tf.strings.length(traj[key][subkey]) != 0

            # All other keys should not be treated as padding
            else:
                pad_mask_dict[subkey] = tf.ones([traj_len], dtype=tf.bool)

        traj[key]["pad_mask_dict"] = pad_mask_dict

    return traj
