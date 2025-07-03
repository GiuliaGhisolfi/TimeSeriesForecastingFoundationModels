import argparse

import pandas as pd
import torch
from autogluon.timeseries.dataset import \
    TimeSeriesDataFrame  # ChronosFineTuningDataset
from autogluon.timeseries.models import ChronosModel
from datasets import Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from uni2ts.loss.packed import PackedNLLLoss

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
    # Load HuggingFace dataset from disk
    hf_dataset = Dataset.load_from_disk("data/moirai_dataset_splitted")
    print("HuggingFace dataset loaded")

    # Convert HuggingFace dataset to pandas DataFrame
    rows = []
    i = 0
    for entry in tqdm(hf_dataset, desc="Preparing dataset"):
        if i == 20: # FIXME
            break

        item_id = entry["item_id"]
        start = pd.to_datetime(entry["start"])
        freq = entry["freq"]
        freq = freq.replace("T", "min").replace("H", "h")
        target_values = entry["target"][0]  # outer sequence -> single time series
        timestamps = pd.date_range(start=start, periods=len(target_values), freq=freq)

        for t, val in zip(timestamps, target_values):
            rows.append({
                "item_id": item_id,
                "timestamp": t,
                "target": val,
            })

        i += 1

    df = pd.DataFrame(rows)

    # Convert to AutoGluon TimeSeriesDataFrame
    ts_df = TimeSeriesDataFrame.from_data_frame(
        df,
    )
    print("Dataset ready")

    # Define context and prediction lengths
    context_length = 2048
    prediction_length = 256
    min_required_length = context_length + prediction_length

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

    # Fit the model with train and validation data
    model.fit(
        train_data=train_df,
        tuning_data=val_df
    )

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
        model_path=args.model_name,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
