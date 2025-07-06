import os
from datetime import timedelta

import numpy as np
from datasets import Dataset, Features, Sequence, Value
from tqdm import tqdm


def split_long_series_dataset(
    indexed_dataset: Dataset,
    context_length: int = 2048,
    prediction_length: int = 256,
    stride: int = None,
) -> Dataset:
    """
    Splits time series in indexed_dataset into multiple sequences of length context + prediction.
    """
    total_length = context_length + prediction_length
    stride = stride or prediction_length  # overlap by default

    new_samples = []

    # define features
    features = Features({
            "item_id": Value("string"),
            "start": Value("timestamp[ns]"),
            "freq": Value("string"),
            "target": Sequence(Sequence(Value("float32"))),
            "dataset": Value("string"),
        })

    for sample in tqdm(indexed_dataset, desc="Splitting time series"):
        ts = np.array(sample["target"], dtype=np.float32)  # shape: (time, n_dims)

        if ts.ndim == 1:
            ts = ts[:, None] # shape: (time, 1)
        elif ts.ndim == 2 and ts.shape[1] > 1:
            ts = ts[:, :1]  # shape: (time, 1)
        elif ts.ndim == 2 and ts.shape[1] == 1:
            pass

        ts_len, n_dims = ts.shape
        num_slices = max((ts_len - total_length) // stride + 1, 0)

        if num_slices == 0:
            new_samples.append({
                "target": sample["target"],
                "item_id": sample["item_id"],
                "start": sample["start"],
                "freq": sample["freq"],
                "dataset": sample["dataset"],
            })
        else:
            for i in range(num_slices):
                start_idx = i * stride
                end_idx = start_idx + total_length

                sliced_target = ts[start_idx:end_idx, :] # shape: (total_length, 1)

                new_samples.append({
                    "target": sliced_target.tolist(),
                    "item_id": sample["item_id"],
                    "start": sample["start"],
                    "freq": sample["freq"],
                    "dataset": sample["dataset"],
                })

    print(f"Created {len(new_samples)} sliced samples.")
    full_dataset = Dataset.from_list(new_samples, features=features)

    return full_dataset
