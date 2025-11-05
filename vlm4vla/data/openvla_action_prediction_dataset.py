from logging.config import dictConfig
from typing import Any, Dict

from vlm4vla.data.base_openvla_dataset import RLDSDataset
from vlm4vla.data.base_action_prediction_dataset import ActionPredictionDataset


class OpenVLADataset(ActionPredictionDataset, RLDSDataset):

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        ActionPredictionDataset.__init__(self, **kwargs)
        if self.organize_type == "interleave":
            kwargs["window_sample"] = "sliding"
            kwargs["left_pad"] = False
        elif self.organize_type == "segment":
            kwargs["window_sample"] = "range"
            kwargs["left_pad"] = True
        else:
            raise ValueError("organize type must be interleave or segment")
        kwargs["chunk_action"] = True
        RLDSDataset.__init__(self, **kwargs)

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in RLDSDataset.__iter__(self):
            yield self.batch_transform(
                task_description=rlds_batch["task"]["language_instruction"].decode(),
                action=rlds_batch["action"],
                episode_mask=rlds_batch["chunk_mask"],
                images=rlds_batch["observation"]["image_primary"],
                gripper_images=None,
            )
