import numpy as np
import torch
from torch.utils.data import default_collate, default_convert

from moirai_utils.moirai_utils import pad_tensor
from uni2ts.common.typing import BatchedSample, Sample
from uni2ts.data.loader import PackCollate, PadCollate


class CostumPadCollate(PadCollate):

    def pad_samples(self, batch: list[Sample]) -> BatchedSample:
        # num features (variate)
        max_feat_dim = 1
        for sample in batch:
            tensor = sample[self.target_field]
            if tensor.ndim == 1:
                feat_dim = 1
            else:
                feat_dim = tensor.shape[1]
            max_feat_dim = max(max_feat_dim, feat_dim)

        # Padding
        for sample in batch:
            length = len(sample[self.target_field])
            feat_dim = sample[self.target_field].shape[1]

            # Fix "observed_mask" shape
            #FIXME: se viene salvato dataset nel formato corretto togliere questa riga
            #sample["observed_mask"] = sample["observed_mask"].unsqueeze(-1)

            for key in self.seq_fields:
                if sample[key].dim() == 1:
                    sample[key] = sample[key].unsqueeze(-1)

                print(key, sample[key].shape)
                # Padding ts length
                sample[key] = torch.cat(
                    [
                        default_convert(sample[key]),
                        default_convert(
                            self.pad_func_map[key](
                                shape=(self.max_length - length,) + sample[key].shape[1:],
                                dtype=sample[key].dtype,
                            )
                        ),
                    ]
                )

                # Padding features dim
                sample[key] = torch.cat(
                    [
                        default_convert(sample[key]),
                        default_convert(
                            self.pad_func_map[key](
                                shape=(self.max_length, max_feat_dim-feat_dim),
                                dtype=sample[key].dtype,
                            )
                        ),
                    ],
                    dim = -1
                )

        return default_collate(batch)