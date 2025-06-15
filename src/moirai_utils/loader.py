import numpy as np
import torch
from torch.utils.data import default_collate, default_convert
from torch.utils.data._utils.collate import default_convert

from moirai_utils.moirai_utils import pad_tensor
from uni2ts.common.typing import BatchedSample, Sample
from uni2ts.data.loader import PackCollate, PadCollate


def to_tensor_safe(value): # FIXME: idk
    if isinstance(value, torch.Tensor):
        return value
    elif isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    elif isinstance(value, (float, int)):
        return torch.tensor(value)
    elif isinstance(value, str):
        return torch.tensor([ord(c) for c in value], dtype=torch.int32)  # string → tensor of char codes
    elif isinstance(value, np.datetime64):
        return torch.tensor([value.astype("datetime64[s]").astype(int)])
    elif isinstance(value, list):
        # solo se lista di valori numerici
        if all(isinstance(x, (int, float)) for x in value):
            return torch.tensor(value)
    raise TypeError(f"Cannot convert value of type {type(value)} to tensor")

class CostumPadCollate(PadCollate):
    def  __call__(self, batch: list[Sample]) -> BatchedSample:
        # Custom padding logic if needed
        return super().__call__(batch)

    def pad_samples(self, batch):
        processed_batch = []
        for sample in batch:
            padded_sample = {}
            length = len(sample[self.target_field])
            for key, value in sample.items():
                tensor_value = to_tensor_safe(value)
                if key in self.seq_fields:
                    pad_size = (self.max_length - length,) + tensor_value.shape[1:]
                    pad_tensor_fn = self.pad_func_map.get(key, pad_tensor)
                    padded_value = torch.cat([tensor_value, pad_tensor_fn(pad_size, tensor_value.dtype)])
                    padded_sample[key] = padded_value
                else:
                    padded_sample[key] = tensor_value
            processed_batch.append(padded_sample)
        return default_collate(processed_batch)
