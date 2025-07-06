import os
from datetime import timedelta

import numpy as np
from datasets import (Dataset, Features, Sequence, Value, concatenate_datasets,
                      load_from_disk)
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
    samples_count = 0
    chunk_size = 100000
    num_chunks = 0

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
            samples_count += 1
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
                samples_count += 1

        if samples_count >= chunk_size:
            dataset_chunk = Dataset.from_list(new_samples, features=features)
            dataset_chunk.save_to_disk(f"data/split_part_{num_chunks}.arrow")

            samples_count = 0
            num_chunks += 1
            new_samples = []

    if samples_count != 0:
        dataset_chunk = Dataset.from_list(new_samples, features=features)
        dataset_chunk.save_to_disk(f"data/split_part_{num_chunks}.arrow")
        num_chunks += 1

    print(f"Created {len(new_samples)} sliced samples.")

    # Save dataset
    all_chunks = [load_from_disk(f"data/split_part_{i}.arrow") for i in range(num_chunks)]
    full_dataset = concatenate_datasets(all_chunks)
    #full_dataset.save_to_disk("data/final_split_dataset")

    """for i in range(num_chunks):
        os.system(f"rm -rf data/split_part_{i}.arrow")"""

    return full_dataset
