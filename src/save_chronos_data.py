from collections import defaultdict

import numpy as np
import pandas as pd
from datasets import load_from_disk
from tqdm import tqdm


def pad_series(ts, required_len):
    pad_len = required_len - len(ts)
    if pad_len > 0:
        padding = [np.nan] * pad_len
        ts = padding + list(ts)
    return ts

def hf_to_dataframe(dataset, min_required_length):
    rows = []
    for entry in tqdm(dataset, desc="Preparing dataset"):
        item_id = entry["item_id"]
        start = pd.to_datetime(entry["start"], errors="coerce")  # Coerce invalid
        freq = entry["freq"].replace("T", "min").replace("H", "h")  # Standardize

        # Skip rows with invalid start
        if pd.isna(start):
            continue

        values = pad_series(entry["target"][0], min_required_length)

        try:
            timestamps = pd.date_range(start=start, periods=len(values), freq=freq)
        except (pd.errors.OutOfBoundsDatetime, OverflowError):
            print(f"Skipping entry with item_id={item_id}, start={start}, freq={freq}")
            continue

        for t, val in zip(timestamps, values):
            rows.append({
                "item_id": item_id,
                "timestamp": t,
                "target": val,
            })

    return pd.DataFrame(rows)

def main():
    context_length = 512 #2048
    prediction_length = 256
    num_chunks = 8

    for i in range(6, num_chunks):
        print(f"Processing split_part_{i}.arrow")

        # Load HuggingFace dataset from disk
        dataset = load_from_disk(f"data/split_part_{i}.arrow")
        
        # Convert HuggingFace dataset to pandas DataFrame
        df = hf_to_dataframe(dataset, min_required_length=context_length+prediction_length)
        df.to_parquet(f"data/chronos_parquet_splits/split_{i}.parquet", index=False)
    
    print("Done :)")

if __name__ == "__main__":
    main()