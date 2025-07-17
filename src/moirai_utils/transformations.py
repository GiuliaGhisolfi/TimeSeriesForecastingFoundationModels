from typing import Any

import numpy as np
import torch

from uni2ts.transform import (AddObservedMask, AddTimeIndex, AddVariateIndex,
                              Transformation, GetPatchSize)

class ToTorch(Transformation):
    def __init__(self, prediction_length=256):
        self.prediction_length = prediction_length

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        # Convert target to torch tensor
        target = torch.tensor(data_entry["target"], dtype=torch.float32)
        if target.ndim == 2:
            target = target[:, 0].unsqueeze(-1)  # Ensure target is 2D with shape (time, 1)        
        data_entry["target"] = target

        # AddObservedMask
        data_entry["observed_mask"] = torch.ones_like(data_entry["target"], dtype=torch.bool)

        # AddTimeIndex
        data_entry["time_id"] = torch.zeros(
                target.shape[:-1], dtype=torch.long, device=target.device
            )

        # AddVariateIndex
        data_entry["variate_id"] = torch.zeros(
                target.shape[:-1], dtype=torch.long, device=target.device
            )

        # Set prediction mask
        prediction_mask = torch.zeros(
                target.shape[:-1], dtype=torch.bool, device=target.device
            )
        prediction_mask[-self.prediction_length:] = True
        data_entry["prediction_mask"] = prediction_mask

        # Set patch size
        patch_size_transform = GetPatchSize(min_time_patches=1) # 16
        data_entry = patch_size_transform(data_entry)

        return {
            "target": data_entry["target"],
            "observed_mask": data_entry["observed_mask"],
            "time_id": data_entry["time_id"],
            "variate_id": data_entry["variate_id"],
            "prediction_mask": data_entry["prediction_mask"],
            "patch_size": data_entry["patch_size"]
        }