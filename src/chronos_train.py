import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from autogluon.timeseries.dataset import \
    TimeSeriesDataFrame  # ChronosFineTuningDataset
from autogluon.timeseries.models import ChronosModel
from datasets import Dataset, concatenate_datasets, load_from_disk
from sklearn.model_selection import train_test_split
from tqdm import tqdm

MODEL_NAME = "chronos-bolt-tiny" # "chronos-bolt-mini", "chronos-bolt-small", "chronos-bolt-base"
MODEL_MAP = {
    "chronos-bolt-tiny": "autogluon/chronos-bolt-tiny",
    "chronos-bolt-mini": "autogluon/chronos-bolt-mini",
    "chronos-bolt-small": "autogluon/chronos-bolt-small",
    "chronos-bolt-base": "autogluon/chronos-bolt-base",
}

EPOCHS = 10
TEST_SIZE = 0.2
PATIENCE = 3
RANDOM_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def train(
    model_name=MODEL_NAME,
    device=DEVICE,
    epochs=EPOCHS,
    patience=PATIENCE,
    test_size=TEST_SIZE,
    batch_size=16,
    learning_rate=1e-5,
    #eval_metric=, (str) #TODO: ?
    ):

    context_length = 512 #2048
    prediction_length = 256
    num_chunks = 8

    if not all(os.path.exists(f"data/chronos_parquet_splits/split_{i}.parquet") for i in range(num_chunks)):
        os.makedirs("data/chronos_parquet_splits", exist_ok=True)

        for i in range(num_chunks):
            print(f"Processing split_part_{i}.arrow")

            # Load HuggingFace dataset from disk
            dataset = load_from_disk(f"data/split_part_{i}.arrow")
            
            # Convert HuggingFace dataset to pandas DataFrame
            df = hf_to_dataframe(dataset, min_required_length=context_length+prediction_length)
            df.to_parquet(f"data/chronos_parquet_splits/split_{i}.parquet", index=False)

    # Concatenate all datasets
    df_list = [
        pd.read_parquet(f"data/chronos_parquet_splits/split_{i}.parquet")
        for i in range(num_chunks)
    ]
    full_df = pd.concat(df_list, ignore_index=True)

    ts_df = TimeSeriesDataFrame(full_df)
    print("Dataset ready")

    # Split item_ids into train and validation sets (80/20 split)
    unique_ids = ts_df.item_ids
    train_ids, val_ids = train_test_split(unique_ids, test_size=test_size, random_state=RANDOM_SEED)

    # Select time series by ID
    train_df = ts_df.loc[train_ids]
    val_df = ts_df.loc[val_ids]

    # Calculate number of steps per epoch
    num_train_samples = len(train_df)
    steps_per_epoch = max(1, num_train_samples // batch_size)

    # Total fine tuning steps
    total_fine_tune_steps = epochs * steps_per_epoch

    # Initialize ChronosModel with fine-tuning, early stopping, and device set
    model_path = MODEL_MAP[model_name]

    hyperparameters = {
        "path": model_path,
        "context_length": context_length,
        "prediction_length": prediction_length,
        "fine_tune": True,
        "fine_tune_lr": learning_rate,
        "fine_tune_steps": total_fine_tune_steps,
        "fine_tune_batch_size": batch_size,
        "eval_every_step": True,
        "save_checkpoints": True,
        "checkpoint_dir": "checkpoints/",
        "early_stopping_patience": patience,
        "device": device,
    }
    model = ChronosModel(
        name=model_name,
        path=model_path,
        prediction_length=prediction_length,
        hyperparameters=hyperparameters,
    )
    print("Model loaded.")
    print(model)

    # Fit the model with train and validation data
    """model.fit(
        train_data=train_df,
        tuning_data=val_df
    )"""

    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-5)

    args = parser.parse_args()

    train(
        model_name=args.model_name,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
