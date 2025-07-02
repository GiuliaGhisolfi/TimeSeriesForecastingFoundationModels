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

    for sample in tqdm(indexed_dataset, desc="Splitting time series"):
        ts = np.array(sample["target"], dtype=np.float32)  # shape: (n_dims, time)

        # to univariate
        if ts.ndim == 2 and ts.shape[0] > 1:
            ts = ts[0:1]  # shape: (1, time)
        elif ts.ndim == 1:
            ts = ts[None, :]  # shape: (1, time)

        n_dims, ts_len = ts.shape
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

                sliced_target = ts[:, start_idx:end_idx]

                # compute new timestamp
                freq_str = sample["freq"]
                start_time = sample["start"]

                if isinstance(start_time, str):
                    start_time = np.datetime64(start_time)

                if "min" in freq_str:
                    minutes = int(freq_str.split()[0])
                    freq = np.timedelta64(minutes, 'm')
                elif freq_str in ["H", "1H"]:
                    freq = np.timedelta64(1, 'h')
                elif freq_str == "6H":
                    freq = np.timedelta64(6, 'h')
                elif freq_str.upper().startswith("D"):
                    freq = np.timedelta64(1, 'D')
                elif freq_str.upper().startswith("W"):
                    freq = np.timedelta64(7, 'D')
                elif freq_str.upper().startswith("M"):
                    freq = np.timedelta64(30, 'D')  # approx
                elif freq_str.upper().startswith("Q"):
                    freq = np.timedelta64(91, 'D')  # approx
                elif freq_str.upper().startswith("A"):
                    freq = np.timedelta64(365, 'D')  # approx
                elif freq_str.upper().endswith("S"):
                    # e.g., W-SUN
                    freq = np.timedelta64(7, 'D')
                else:
                    freq = np.timedelta64(1, 'D')  # fallback

                try:
                    new_start_time = np.datetime64(start_time) + start_idx * freq
                except Exception as e:
                    print(f"Skipping time shift at slice {i} for item {sample['item_id']} — reason: {e}")
                    new_start_time = np.datetime64(start_time)  # fallback

                new_samples.append({
                    "target": [[float(v)] for v in sliced_target[0]],
                    "item_id": sample["item_id"],
                    "start": new_start_time,
                    "freq": sample["freq"],
                    "dataset": sample["dataset"],
                })

    # define features
    features = Features({
            "item_id": Value("string"),
            "start": Value("timestamp[ns]"),
            "freq": Value("string"),
            "target": Sequence(Sequence(Value("float32"))),
            "dataset": Value("string"),
        })

    print(f"Created {len(new_samples)} sliced samples.")
    return Dataset.from_list(new_samples, features=features)
