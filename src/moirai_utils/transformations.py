from typing import Any

import torch

from uni2ts.transform import Transformation


class ToTorch(Transformation):
    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        data_entry["target"] = torch.stack(
            [torch.tensor(dim, dtype=torch.float32) for dim in data_entry["target"]],
            dim=0  # output: (n_dims, time)
        )
        return data_entry
