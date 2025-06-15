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
                if key in self.seq_fields:
                    # Pad sequenze
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
                    # Copia altri campi, evitando oggetti non compatibili
                    if isinstance(value, (str, int, float, torch.Tensor)):
                        padded_sample[key] = value
                    elif isinstance(value, np.generic):
                        # Converti np.datetime64, np.str_, etc.
                        padded_sample[key] = value.item()
                    else:
                        # Scarta se non è gestibile
                        print(f"Ignoring key '{key}' of unsupported type: {type(value)}")
            processed_batch.append(padded_sample)

        return default_collate(processed_batch)

