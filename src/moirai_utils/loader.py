import numpy as np
import torch
from torch.utils.data import default_collate, default_convert

from moirai_utils.moirai_utils import pad_tensor
from uni2ts.common.typing import BatchedSample, Sample
from uni2ts.data.loader import PackCollate, PadCollate


class CostumPadCollate(PadCollate):

    def __init__(self, *args, max_sequence_length, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_sequence_length = max_sequence_length
    
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

        """
        # Trimming
        for sample in batch:
            current_length = len(sample[self.target_field])
            for key in self.seq_fields:
                if current_length > self.max_length:
                    sample[key] = sample[key][:self.max_length]
        """

        # Padding
        for sample in batch:
            current_length = len(sample[self.target_field])
            if current_length > self.max_sequence_length: # Trimming
                for key in self.seq_fields:
                    sample[key] = sample[key][:self.max_sequence_length]
            
            length = len(sample[self.target_field])
            feat_dim = sample[self.target_field].shape[1]
            
            for key in self.seq_fields:
                if sample[key].dim() == 1:
                    sample[key] = sample[key].unsqueeze(-1)

                # Padding ts length
                if length < self.max_sequence_length:
                    sample[key] = torch.cat(
                        [
                            default_convert(sample[key]),
                            default_convert(
                                self.pad_func_map[key](
                                    shape=(self.max_sequence_length - length,) + sample[key].shape[1:],
                                    dtype=sample[key].dtype,
                                )
                            ),
                        ]
                    )

                # Padding features dim
                if max_feat_dim > feat_dim:
                    sample[key] = torch.cat(
                        [
                            default_convert(sample[key]),
                            default_convert(
                                self.pad_func_map[key](
                                    shape=(self.max_sequence_length, max_feat_dim - feat_dim),
                                    dtype=sample[key].dtype,
                                )
                            ),
                        ],
                        dim=-1
                    )

        return default_collate(batch)
    
    def get_sample_id(self, batch):
        sample_id = torch.stack(
            [
                torch.cat([torch.ones(length), torch.zeros(self.max_sequence_length - length)])
                for sample in batch
                if (length := len(sample[self.target_field][:self.max_sequence_length]))
            ]
        ).to(torch.long)
        return sample_id