from typing import Any

import numpy as np
import torch

from uni2ts.transform import (AddObservedMask, AddTimeIndex, AddVariateIndex,
                              Transformation)


class ToTorch(Transformation):
    def __init__(self, context_length=2048, prediction_length=256): #FIX ME: ???
        #super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        total_length = self.context_length + self.prediction_length

        # AddObservedMask
        observed_mask_transform = AddObservedMask(fields=("target",))
        data_entry = observed_mask_transform(data_entry)
        data_entry["observed_mask"] = torch.tensor(data_entry["observed_mask"], dtype=torch.bool).squeeze()

        # Convert target to torch tensor and crop to last total_length
        target = torch.stack(
            [torch.tensor(dim, dtype=torch.float32) for dim in data_entry["target"]],
            dim=0  # (n_dims, time)
        )
        target = target[:, -total_length:]
        data_entry["target"] = target

        # AddTimeIndex
        time_id_transform = AddTimeIndex(fields=("target",))
        data_entry = time_id_transform(data_entry)
        data_entry["time_id"] = torch.tensor(data_entry["time_id"], dtype=torch.long).squeeze()[-total_length:]

        # AddVariateIndex
        variate_id_transform = AddVariateIndex(fields=("target",), max_dim=target.shape[0])
        data_entry = variate_id_transform(data_entry)
        data_entry["variate_id"] = torch.tensor(data_entry["variate_id"], dtype=torch.long).squeeze()[-total_length:]

        # Set prediction mask
        prediction_mask = torch.zeros_like(target, dtype=torch.bool)
        prediction_mask[:, -self.prediction_length:] = True
        data_entry["prediction_mask"] = prediction_mask

        # Set patch size
        data_entry["patch_size"] = torch.tensor(target.shape[1], dtype=torch.long).squeeze()

        return {
            "target": data_entry["target"],
            "observed_mask": data_entry["observed_mask"][-total_length:],
            "time_id": data_entry["time_id"],
            "variate_id": data_entry["variate_id"],
            "prediction_mask": data_entry["prediction_mask"],
            "patch_size": data_entry["patch_size"]
        }