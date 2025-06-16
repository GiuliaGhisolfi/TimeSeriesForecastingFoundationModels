from typing import Any

import numpy as np
import torch

from uni2ts.transform import Transformation


class ToTorch(Transformation):
    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        data_entry["target"] = torch.stack(
            [torch.tensor(dim, dtype=torch.float32) for dim in data_entry["target"]],
            dim=0  # output: (n_dims, time)
        )

        data_entry["item_id"] = str(data_entry["item_id"])
        data_entry["start"] = torch.tensor(int(data_entry["start"].astype('datetime64[s]').astype(int)))
        data_entry["freq"] = str(data_entry["freq"])
        data_entry["dataset"] = str(data_entry["dataset"])

        return data_entry