import numpy as np
import torch
from torch.utils.data import default_collate, default_convert

from moirai_utils.moirai_utils import pad_tensor
from uni2ts.common.typing import BatchedSample, Sample
from uni2ts.data.loader import PackCollate, PadCollate


class CostumPadCollate(PadCollate):
    def  __call__(self, batch: list[Sample]) -> BatchedSample:
        # Custom padding logic if needed
        return super().__call__(batch)
    
    def pad_samples(self, batch: list[Sample]) -> BatchedSample:
        processed_batch = []
        for i, sample in enumerate(batch):
            padded_sample = {}
            length = len(sample[self.target_field])

            for key, value in sample.items():
                if key in self.seq_fields: # target
                    # Pad sequence
                    padded_sample[key] = torch.cat([
                        default_convert(value),
                        default_convert(
                            self.pad_func_map.get(key, pad_tensor)(
                                (self.max_length - length,) + value.shape[1:],
                                value.dtype,
                            )
                        ),
                    ])
                else:
                    if isinstance(value, torch.Tensor):
                        padded_sample[key] = value
                    elif isinstance(value, (int, float)):
                        padded_sample[key] = torch.tensor(value)
                    elif isinstance(value, str):
                        codes = [ord(c) for c in value]
                        padded_sample[key] = torch.tensor(codes, dtype=torch.int32)
                    elif isinstance(value, np.str_):
                        s = value.item()
                        codes = [ord(c) for c in s]
                        padded_sample[key] = torch.tensor(codes, dtype=torch.int32)
                    elif isinstance(value, np.ndarray): # start
                        ts = int(value.astype('datetime64[s]').astype(int))
                        padded_sample[key] = torch.tensor(ts)
                    elif isinstance(value, np.integer) or isinstance(value, np.floating):
                        padded_sample[key] = torch.tensor(value.item())
                    else:
                        print(f"Ignoring key '{key}' of unsupported type: {type(value)}")
            processed_batch.append(padded_sample)

        return default_collate(processed_batch)
