import numpy as np
import torch
from torch.utils.data import default_collate, default_convert

from moirai_utils.moirai_utils import pad_tensor
from uni2ts.common.typing import BatchedSample, Sample
from uni2ts.data.loader import PackCollate, PadCollate


class CostumPadCollate(PadCollate):

    def pad_samples(self, batch: list[Sample]) -> BatchedSample:
        max_feat_dim = max(
            sample[self.target_field].shape[1]
            if sample[self.target_field].ndim > 1
            else 1
            for sample in batch
        )

        for sample in batch:
            length = len(sample[self.target_field])

            for key in self.seq_fields:
                seq = sample[key]

                # Se [L] → [L, 1]
                if seq.ndim == 1:
                    seq = seq.unsqueeze(1)

                # Se [L, d] → padding a [L, max_feat_dim]
                feat_dim = seq.shape[1]
                if feat_dim < max_feat_dim:
                    pad_feat = torch.zeros((length, max_feat_dim - feat_dim), dtype=seq.dtype)
                    seq = torch.cat([seq, pad_feat], dim=1)

                # Padding a max_length in lunghezza
                if length < self.max_length:
                    pad_seq = torch.zeros((self.max_length - length, max_feat_dim), dtype=seq.dtype)
                    seq = torch.cat([seq, pad_seq], dim=0)

                sample[key] = seq

        return default_collate(batch)
