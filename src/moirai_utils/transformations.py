from typing import Any

import numpy as np
import torch

from uni2ts.transform import (AddObservedMask, AddTimeIndex, AddVariateIndex,
                              Transformation)


class ToTorch(Transformation):
    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        # AddObservedMask
        observed_mask_transform = AddObservedMask(fields=("target",))
        data_entry = observed_mask_transform(data_entry)
        data_entry["observed_mask"] = torch.tensor(data_entry["observed_mask"], dtype=torch.bool)

        # Convert target to torch tensor
        data_entry["target"] = torch.stack(
            [torch.tensor(dim, dtype=torch.float32) for dim in data_entry["target"]],
            dim=0  # output: (n_dims, time)
        )
    
        # AddTimeIndex
        time_id_transform = AddTimeIndex(fields=("target",))
        data_entry = time_id_transform(data_entry)
        data_entry["time_id"] = torch.tensor(data_entry["time_id"], dtype=torch.long)

        # AddVariateIndex
        variate_id_transform = AddVariateIndex(fields=("target",), max_dim=data_entry["target"].shape[0])
        data_entry = variate_id_transform(data_entry)
        data_entry["variate_id"] = torch.tensor(data_entry["variate_id"], dtype=torch.long)

        data_entry["prediction_mask"] = torch.ones_like(data_entry["target"], dtype=torch.bool)

        data_entry["patch_size"] = torch.tensor(data_entry["target"].shape[1], dtype=torch.long)

        return {
            "target": data_entry["target"],
            "observed_mask": data_entry["observed_mask"],
            "time_id": data_entry["time_id"],
            "variate_id": data_entry["variate_id"],
            "prediction_mask": data_entry["prediction_mask"],
            "patch_size": data_entry["patch_size"]
        }