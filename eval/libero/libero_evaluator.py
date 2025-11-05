import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import tqdm
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

import imageio


def run_libero_eval_single_episode(
    env,
    task_description,
    obs,
    max_steps,
    video_name,
    model,
    ckpt_path,
    model_name,
    logging_dir,
    num_steps_wait=10,
    save_video=False,
):

    # Initialize logging

    # Initialize model
    model.reset()
    timestep = 0
    images = []
    predicted_actions = []

    # Step the environment
    while timestep < max_steps + num_steps_wait:
        if timestep < num_steps_wait:
            # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
            # and we need to wait for them to fall
            obs, reward, done, info = env.step([0, 0, 0, 0, 0, 0, -1])
            timestep += 1
        image = get_libero_image(obs, 224)
        images.append(image)
        action = model.step(image, task_description)
        predicted_actions.append(action)
        # step the environment
        obs, reward, done, info = env.step(action.tolist())

        success = "success" if done else "failure"
        if done:
            break
        timestep += 1

    # save video
    if save_video:
        ckpt_path_basename = ckpt_path if ckpt_path[-1] != "/" else ckpt_path[:-1]
        ckpt_path_basename = ckpt_path_basename.split("/")[-1]
        ckpt_path_basename = f"{model_name}_{ckpt_path_basename}"
        video_name = f"{success}_{video_name}"
        # video_path = f"{ckpt_path_basename}/{video_name}"
        video_path = os.path.join(logging_dir, video_name)
        video_writer = imageio.get_writer(video_path, fps=40)
        for img in images:
            video_writer.append_data(img)
        video_writer.close()

        # save action trajectory
        action_path = video_path.replace(".mp4", ".png")
        action_root = os.path.dirname(action_path) + "/actions/"
        os.makedirs(action_root, exist_ok=True)
        action_path = action_root + os.path.basename(action_path)
        model.visualize_epoch(predicted_actions, images, save_path=action_path)

    return success == "success"


def get_libero_env(task, resolution=256):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def get_libero_image(obs, resize_size=224):
    """Extracts image from observations and preprocesses it."""
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing

    # def resize_image(img, resize_size):
    # import tensorflow as tf
    #     """
    #     Takes numpy array corresponding to a single image and returns resized image as numpy array.

    #     NOTE (Moo Jin): To make input images in distribution with respect to the inputs seen at training time, we follow
    #                     the same resizing scheme used in the Octo dataloader, which OpenVLA uses for training.
    #     """
    #     assert isinstance(resize_size, tuple)
    #     # Resize to image size expected by model
    #     img = tf.image.encode_jpeg(img)  # Encode as JPEG, as done in RLDS dataset builder
    #     img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Immediately decode back
    #     img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    #     img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    #     img = img.numpy()
    #     return img

    # img = resize_image(img, resize_size)
    return img


def libero_evaluator(model, args):
    success_arr = []
    kwargs = dict(
        model=model,
        ckpt_path=args.ckpt_path,
        model_name=args.model_name,
        logging_dir=args.logging_dir,
        num_steps_wait=args.num_steps_wait)
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {args.task_suite_name}, num_tasks_in_suite: {num_tasks_in_suite}")
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)
        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)
        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, resolution=256)
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])
            if args.task_suite_name == "libero_spatial":
                max_steps = 220  # longest training demo has 193 steps
            elif args.task_suite_name == "libero_object":
                max_steps = 280  # longest training demo has 254 steps
            elif args.task_suite_name == "libero_goal":
                max_steps = 300  # longest training demo has 270 steps
            elif args.task_suite_name == "libero_10":
                max_steps = 520  # longest training demo has 505 steps
            elif args.task_suite_name == "libero_90":
                max_steps = 400  # longest training demo has 373 steps
            print(f"\nEpisode: {episode_idx+1}, Task: {task_description}")
            video_name = f"{task_description}_taskid_{task_id}_episode_{episode_idx+1}.mp4"
            # INSERT_YOUR_CODE
            # 均匀地保存5个episode的视频
            save_indices = np.linspace(0, args.num_trials_per_task - 1, 5, dtype=int)
            save_video = episode_idx in save_indices
            success_arr.append(
                run_libero_eval_single_episode(
                    env, task_description, obs, max_steps, video_name, save_video=save_video, **kwargs))

    return success_arr