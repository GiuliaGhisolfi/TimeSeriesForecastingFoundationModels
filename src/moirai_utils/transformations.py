from typing import Any

import numpy as np
import torch

from uni2ts.transform import (AddObservedMask, AddTimeIndex, AddVariateIndex,
                              Transformation, GetPatchSize)


class ToTorch(Transformation):
    def __init__(self, context_length=2048, prediction_length=256): #FIXME: ???
        #super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        """# AddObservedMask
        observed_mask_transform = AddObservedMask(fields=("target",))
        data_entry = observed_mask_transform(data_entry)
        #observed_mask = torch.ones_like(target, dtype=torch.bool)
        data_entry["observed_mask"] = torch.tensor(data_entry["observed_mask"], dtype=torch.bool).squeeze()#[:, -total_length:]"""

        # Convert target to torch tensor
        data_entry["target"] = torch.tensor(data_entry["target"])#, dtype=torch.long)

        target = data_entry["target"]
        max_dim = target.shape[0]

        # AddObservedMask
        data_entry["observed_mask"] = torch.ones_like(data_entry["target"], dtype=torch.bool)

        # AddTimeIndex
        """time_id_transform = AddTimeIndex(fields=("target",))
        data_entry = time_id_transform(data_entry)
        data_entry["time_id"] = torch.tensor(data_entry["time_id"], dtype=torch.long).squeeze()#[:, -total_length:]"""
        data_entry["time_id"] = torch.zeros(
                target.shape[:-1], dtype=torch.long, device=target.device
            )

        # AddVariateIndex
        """variate_id_transform = AddVariateIndex(fields=("target",), max_dim=max_dim)
        data_entry = variate_id_transform(data_entry)
        data_entry["variate_id"] = torch.tensor(data_entry["variate_id"], dtype=torch.long).squeeze()#[:, -total_length:]"""
        data_entry["variate_id"] = torch.zeros(
                target.shape[:-1], dtype=torch.long, device=target.device
            )

        # Set prediction mask
        #prediction_mask = torch.zeros_like(data_entry["target"], dtype=torch.bool)
        prediction_mask = torch.zeros(
                target.shape[:-1], dtype=torch.bool, device=target.device
            )
        prediction_mask[-self.prediction_length:] = True
        data_entry["prediction_mask"] = prediction_mask

        # Set patch size
        patch_size_transform = GetPatchSize(min_time_patches=1) # 16
        data_entry = patch_size_transform(data_entry)
        #data_entry["patch_size"] = torch.tensor(data_entry["target"].shape[1], dtype=torch.long) #use patch

        return {
            "target": data_entry["target"],
            "observed_mask": data_entry["observed_mask"],
            "time_id": data_entry["time_id"],
            "variate_id": data_entry["variate_id"],
            "prediction_mask": data_entry["prediction_mask"],
            "patch_size": data_entry["patch_size"]
        }