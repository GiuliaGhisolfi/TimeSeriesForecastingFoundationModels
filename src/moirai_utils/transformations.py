from typing import Any

import numpy as np
import torch

from uni2ts.transform import Transformation, AddObservedMask, AddTimeIndex, AddVariateIndex

class ToTorch(Transformation):
    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        data_entry["target"] = torch.stack(
            [torch.tensor(dim, dtype=torch.float32) for dim in data_entry["target"]],
            dim=0  # output: (n_dims, time)
        )
        # TODO
        observed_mask = AddObservedMask()
        time_id = AddTimeIndex()
        variate_id = AddVariateIndex()
        prediction_mask = pass
        patch_size = pass

        return {
            "target": data_entry["target"],
            "observed_mask": observed_mask,
            "time_id": time_id,
            "variate_id": variate_id,
            "prediction_mask": prediction_mask,
            "patch_size": patch_size,
            } # data_entry