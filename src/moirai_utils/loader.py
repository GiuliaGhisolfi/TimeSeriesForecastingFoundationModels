import numpy as np
import torch
from torch.utils.data import default_collate, default_convert

from moirai_utils.moirai_utils import pad_tensor
from uni2ts.common.typing import BatchedSample, Sample
from uni2ts.data.loader import PackCollate, PadCollate
from uni2ts.transform import (AddObservedMask, AddTimeIndex, AddVariateIndex,
                              Transformation)


class CostumPadCollate(PadCollate):

    def pad_samples(self, batch: list[Sample]) -> BatchedSample:
        # num features (variate)
        max_feat_dim = 1
        for sample in batch:
            for key in self.seq_fields:
                tensor = sample[key]
                if tensor.ndim == 1:
                    feat_dim = 1
                else:
                    feat_dim = tensor.shape[1]
                max_feat_dim = max(max_feat_dim, feat_dim)

        # Padding
        for sample in batch:
            length = len(sample[self.target_field])
            feat_dim = sample[self.target_field].shape[1]

            for key in self.seq_fields:
                # Padding ts length
                sample[key] = torch.cat(
                    [
                        default_convert(sample[key]),
                        default_convert(
                            self.pad_func_map[key](
                                (self.max_length - length,) + sample[key].shape[1:],
                                sample[key].dtype,
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
                                (self.max_length, max_feat_dim-feat_dim),
                                sample[key].dtype,
                            )
                        ),
                    ],
                    dim = -1
                )

            """# AddObservedMask
            observed_mask_transform = AddObservedMask(fields=("target",))
            sample = observed_mask_transform(sample)
            sample["observed_mask"] = torch.tensor(sample["observed_mask"], dtype=torch.bool)

            # AddTimeIndex
            time_id_transform = AddTimeIndex(fields=("target",))
            sample = time_id_transform(sample)
            sample["time_id"] = torch.tensor(sample["time_id"], dtype=torch.long).squeeze()

            # AddVariateIndex
            variate_id_transform = AddVariateIndex(fields=("target",), max_dim=sample["target"].shape[0])
            sample = variate_id_transform(sample)
            sample["variate_id"] = torch.tensor(sample["variate_id"], dtype=torch.long).squeeze()

            sample["prediction_mask"] = torch.ones_like(sample["target"], dtype=torch.bool).squeeze()

            sample["patch_size"] = torch.tensor(sample["target"].shape[1], dtype=torch.long).squeeze()"""

        return default_collate(batch)