from typing import Optional, Sequence, Callable
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import cv2 as cv
import json
from transforms3d.euler import euler2axangle

from vlm4vla.train.base_trainer import BaseTrainer
from eval.calvin.model_wrapper import CustomModel


class BaseModelInference(CustomModel):

    def __init__(
        self,
        ckpt_path,
        configs,
        device,
        save_dir=None,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "widowx_bridge",
        exec_horizon=1,
        execute_step=1,
    ):
        self.configs = configs
        self.dataset_stat = self.load_dataset_stat()
        self.model = BaseTrainer(configs=configs)
        self.policy = self.model
        self.execute_step = execute_step

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if policy_setup == "widowx_bridge":
            unnorm_key = "bridge_orig" if unnorm_key is None else unnorm_key
        elif policy_setup == "google_robot":
            unnorm_key = "fractal20220817_data" if unnorm_key is None else unnorm_key
        else:
            raise NotImplementedError(
                f"Policy setup {policy_setup} not supported for octo models. The other datasets can be found in the huggingface config.json file."
            )
        self.sticky_gripper_num_repeat = 2

        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key
        if self.policy_setup == "google_robot":
            self.close_gripper_act = -1
        elif self.policy_setup == "widowx_bridge":
            self.close_gripper_act = 1
        else:
            raise NotImplementedError

        print(
            f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key}, execute_step/forward_n: {self.execute_step}/{self.configs['fwd_pred_next_n']}***"
        )
        # self.tokenizer, self.image_processor, self.model = self._init_policy()

        self.image_size = self.configs.get("image_size", 224)
        self.action_scale = self.configs.get("action_scale", 1.0)
        self.horizon = self.configs["window_size"]
        self.window_size = self.horizon
        self.pred_action_horizon = self.configs["fwd_pred_next_n"]
        self.exec_horizon = exec_horizon
        # repeat the closing gripper action for self.sticky_gripper_num_repeat times (google robot setting)
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        self.late_close_gripper = 2
        self.close_gripper_num = 0

        self.task = None
        self.task_description = None
        self.num_image_history = 0

        self.init_config(ckpt_path, configs, device, save_dir)
        self.raw_calvin = True

    def reset(self):
        super().reset()

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.close_gripper_num = 0
        self.previous_gripper_action = None

    @staticmethod
    def load_dataset_stat():
        stat = {}

        with open("configs/data/oxe_dataset_stats/dataset_statistics_google.json", "r") as f:
            google_info = json.load(f)
        stat["fractal20220817_data"] = google_info

        with open("configs/data/oxe_dataset_stats/dataset_statistics_bridge.json", "r") as f:
            bridge_info = json.load(f)
        stat["bridge_orig"] = bridge_info

        return stat

    def transform_action(self, raw_actions):
        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),  # range [0, 1]; 1 = open; 0 = close
        }
        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale
        action_rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
        roll, pitch, yaw = action_rotation_delta
        action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
        action_rotation_axangle = action_rotation_ax * action_rotation_angle
        action["rot_axangle"] = action_rotation_axangle * self.action_scale

        if self.policy_setup == "google_robot":
            current_gripper_action = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
            # current_gripper_action = raw_action["open_gripper"]
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
            else:
                relative_gripper_action = (self.previous_gripper_action - current_gripper_action)
            self.previous_gripper_action = current_gripper_action

            if np.abs(relative_gripper_action) > 0.5 and (not self.sticky_action_is_on):
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action

            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper_action = self.sticky_gripper_action

            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0

            action["gripper"] = relative_gripper_action
            print(f'action gripper: {action["gripper"]}')

        elif self.policy_setup == "widowx_bridge":
            relative_gripper_action = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
            if relative_gripper_action[0] > 0:
                self.close_gripper_num += 1
            else:
                self.close_gripper_num = 0

            if self.close_gripper_num >= self.late_close_gripper:
                relative_gripper_action[0] = 1
            else:
                relative_gripper_action[0] = -1

            action["gripper"] = relative_gripper_action

        action["terminate_episode"] = np.array([0.0])

        return raw_action, action

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(
            image,
            tuple((self.image_size, self.image_size)),
            interpolation=cv.INTER_AREA,
        )
        return image

    def visualize_epoch(
        self,
        predicted_raw_actions: Sequence[np.ndarray],
        images: Sequence[np.ndarray],
        save_path: str,
    ) -> None:
        images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]

        img_strip = np.concatenate(np.array(images[::3]), axis=1)

        # set up plt figure
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        # plot actions
        pred_actions = np.array([
            np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
            for a in predicted_raw_actions
        ])
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            # actions have batch, horizon, dim, in this example we just take the first action for simplicity
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)

    def step(self, image, goal):
        obs = {}
        obs['rgb_obs'] = {}
        obs["rgb_obs"]['rgb_static'] = image
        action = super().step(obs, goal, execute_step=self.execute_step)

        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action)

        if isinstance(action, torch.Tensor):
            action = action.squeeze()
            action = action.reshape(-1, action.shape[-1])
            action = action.numpy()

        action_norm_stats = self.dataset_stat[self.unnorm_key]["action"]
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        action = np.where(
            mask,
            0.5 * (action + 1) * (action_high - action_low) + action_low,
            action,
        )
        raw_action, env_action = self.transform_action(action)

        return raw_action, env_action
