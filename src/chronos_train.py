import argparse
import json
import os
import time
from collections import defaultdict

import dask.dataframe as dd
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



def hf_to_autogluon_df(dataset, min_required_length=64):
    rows = []
    i = 0
    for entry in tqdm(dataset, desc="Preparing dataset"):
        if i == 20:
            break

        item_id = entry.get("id") or entry.get("item_id")
        start = pd.to_datetime(entry["start"], errors="coerce")  # Coerce invalid
        freq = entry["freq"].replace("T", "min").replace("H", "h")  # Standardize

        # Skip rows with invalid start
        if pd.isna(start):
            continue

        raw = entry["target"]
        normalized = [[x[0]] if isinstance(x, list) else [x] for x in raw]
        values = np.array(normalized, dtype=np.float32)
        #values = raw_values[~np.isnan(raw_values)]
        #values = pad_series(values, min_required_length)
        #print(values)

        try:
            timestamps = pd.date_range(start=start, periods=len(values), freq=freq)
        except (pd.errors.OutOfBoundsDatetime, OverflowError):
            print(f"Skipping entry with item_id={item_id}, start={start}, freq={freq}")
            continue

        for t, val in zip(timestamps, values):
            if val is not np.nan:
                rows.append({
                    "item_id": item_id,
                    "timestamp": t,
                    "target": val[0], #np.array([val], dtype=np.float32),
                })
        i += 1

    full_df = pd.DataFrame(rows)
    print(full_df)
    return TimeSeriesDataFrame.from_data_frame(full_df)

def train(
    model_name=MODEL_NAME,
    device=DEVICE,
    epochs=EPOCHS,
    patience=PATIENCE,
    test_size=TEST_SIZE,
    batch_size=2,
    learning_rate=1e-5,
    ):
    context_length = 256 #2048
    prediction_length = 64 #256

    # Concatenate all datasets
    shard_path = "data/moirai_dataset/data-00000-of-00055.arrow"
    ds = Dataset.from_file(shard_path)
    #ds = Dataset.load_from_disk("data/moirai_dataset") # FIXME
    ts_df = hf_to_autogluon_df(ds, min_required_length=context_length+prediction_length)
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
        "fine_tune_trainer_kwargs": {
            "optim": "adamw_torch",
        },
        "fine_tune_lr": learning_rate,
        "fine_tune_steps": total_fine_tune_steps,
        "fine_tune_batch_size": batch_size,
        "eval_every_step": True,
        "save_checkpoints": True,
        "checkpoint_dir": "checkpoints/",
        "early_stopping_patience": patience,
        "device": "auto",
        "logging_strategy": "steps",
        "logging_steps": 50,  # stampa ogni n step
        "report_to": "none",
        "disable_tqdm": False,
        "verbose": True,
    }
    model = ChronosModel(
        name=model_name,
        path=model_path,
        prediction_length=prediction_length,
        hyperparameters=hyperparameters,
    )
    print("Model loaded.")

    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("PyTorch sees CUDA:", torch.cuda.is_available())

    # Fit the model with train and validation data
    print(">> START TRAINING ...")
    start_time = time.time()

    model.fit(
        train_data=train_df,
        tuning_data=val_df
    )

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Training complete in {total_time:.2f} seconds.")

    if hasattr(model.model, "trainer"):
        log_history = model.model.trainer.state.log_history
        train_losses = []
        val_losses = []
        epoch_times = []

        last_time = start_time
        for log in log_history:
            if "loss" in log:
                train_losses.append(log["loss"])
            if "eval_loss" in log:
                val_losses.append(log["eval_loss"])
            if "epoch" in log:
                now = time.time()
                epoch_times.append(now - last_time)
                last_time = now

        # Save to file
        with open(f"results/{model_name}_training_logs.json", "w") as f:
            json.dump({
                "train_loss": train_losses,
                "val_loss": val_losses,
                "epoch_times": epoch_times,
                "total_time_sec": total_time
            }, f, indent=2)

        print("Training logs saved to training_logs.json")
    else:
        print("Trainer logs not found; could not record loss or times.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=1)#EPOCHS) #FIXME
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)

    args = parser.parse_args()

    train(
        model_name=args.model_name,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
