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
import tensorflow as tf


class BaseModelInference(CustomModel):

    def __init__(self,
                 ckpt_path,
                 configs,
                 device,
                 save_dir=None,
                 unnorm_key: Optional[str] = None,
                 policy_setup: str = "widowx_bridge",
                 exec_horizon=1,
                 execute_step=1,
                 center_crop=False):
        self.configs = configs
        self.dataset_stat = self.load_dataset_stat()
        self.model = BaseTrainer(configs=configs)
        self.policy = self.model
        self.execute_step = execute_step
        self.center_crop = center_crop

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if policy_setup == "widowx_bridge":
            unnorm_key = "bridge_orig" if unnorm_key is None else unnorm_key
        elif policy_setup == "google_robot":
            unnorm_key = "fractal20220817_data" if unnorm_key is None else unnorm_key
        elif "libero" in policy_setup:
            unnorm_key = policy_setup + "_no_noops"
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
        elif "libero" in self.policy_setup:
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

        with open("configs/data/oxe_dataset_stats/dataset_statistics_libero_10.json", "r") as f:
            libero_10_info = json.load(f)
        stat["libero_10_no_noops"] = libero_10_info

        return stat

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

        img_strip = np.concatenate(np.array(images[::30]), axis=1)

        # set up plt figure
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        # plot actions
        pred_actions = np.array(predicted_raw_actions)
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
        if self.center_crop:
            batch_size = 1
            crop_scale = 0.9

            # Convert to TF Tensor and record original data type (should be tf.uint8)
            image = tf.convert_to_tensor(np.array(image))
            orig_dtype = image.dtype

            # Convert to data type tf.float32 and values between [0,1]
            image = tf.image.convert_image_dtype(image, tf.float32)

            # Crop and then resize back to original size
            image = crop_and_resize(image, crop_scale, batch_size)

            # Convert back to original data type
            image = tf.clip_by_value(image, 0, 1)
            image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

            # Convert back to PIL Image
            # image = Image.fromarray(image.numpy())
            # image = image.convert("RGB")
            image = image.numpy()

        obs["rgb_obs"]['rgb_static'] = image
        action = super().step(obs, goal, execute_step=self.execute_step)
        # print(action)
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
        # action[..., -1] = np.sign(action[..., -1]) * -1.0  # binarize and invert gripper action
        # action[..., -1] = np.sign(action[..., -1])
        action[..., -1] = -1.0 * (2.0 * (action[..., -1] > 0.5) - 1.0)
        return action.squeeze()


def crop_and_resize(image, crop_scale, batch_size):
    """
    Center-crops an image to have area `crop_scale` * (original image area), and then resizes back
    to original size. We use the same logic seen in the `dlimp` RLDS datasets wrapper to avoid
    distribution shift at test time.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) and datatype tf.float32 with
               values between [0,1].
        crop_scale: The area of the center crop with respect to the original image.
        batch_size: Batch size.
    """
    # Convert from 3D Tensor (H, W, C) to 4D Tensor (batch_size, H, W, C)
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Get height and width of crop
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Get bounding box representing crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Crop and then resize back up
    image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))

    # Convert back to 3D Tensor (H, W, C)
    if expanded_dims:
        image = image[0]

    return image