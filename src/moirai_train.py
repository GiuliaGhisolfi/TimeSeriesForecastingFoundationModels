import argparse
import os
import pickle

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils._pytree import tree_map

from moirai_utils.callbacks import EpochStatsLogger
from moirai_utils.loader import CostumPadCollate
from moirai_utils.moirai_utils import (get_train_and_val_datasets,
                                       pad_bool_tensor, pad_int_tensor,
                                       pad_tensor, to_timeseries_dataset)
from uni2ts.data.loader import DataLoader
from uni2ts.loss.packed import PackedNLLLoss
from uni2ts.model.moirai import MoiraiFinetune, MoiraiModule

torch.set_float32_matmul_precision('medium') # TODO: change
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

MODEL_PATH = "Salesforce/moirai-1.0-R-small"
MODEL_NAME = "moirai_small"
DEVICE_MAP = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 10
TEST_SIZE = 0.2
PATIENCE = 3


def move_batch_to_device(batch, device):
    return tree_map(lambda x: x.to(device) if isinstance(x, torch.Tensor) else x, batch)

def split_long_time_series(dataset_obj: Dataset, max_seq_len: int):
    """
    Splits long time series in a dataset into multiple shorter series,
    each with a maximum length of max_seq_len.
    'target' is sliced.
    'start' is adjusted for each sub-series.
    Other metadata fields are duplicated.
    """
    print(f"Splitting time series to max_seq_len: {max_seq_len}")
    new_data_lists = []

    for sample, target_obj in zip(dataset_obj.indexer.dataset, dataset_obj):
        groups = {}

        target = target_obj["target"]
        current_ts_length = len(target) # Access target directly from the sample dict

        if current_ts_length <= max_seq_len:
            for key in ["item_id", "start", "freq", "dataset"]:
                groups[key] = sample[key]
            groups["target"] = target
        
        else:
            # Split the long series
            num_chunks = (current_ts_length + max_seq_len - 1) // max_seq_len
            
            for k in range(num_chunks):
                start_idx = k * max_seq_len
                end_idx = min((k + 1) * max_seq_len, current_ts_length)

                # Target
                groups["target"] = target[start_idx:end_idx]

                # "Start"
                original_start_timestamp = pd.to_datetime(sample["start"])
                groups["start"] = original_start_timestamp + start_idx * pd.to_timedelta(
                    str(sample["freq"]).replace("T", "min"))

                # Handle other metadata fields by duplicating
                for key in ["item_id", "freq", "dataset"]:
                    groups[key] = sample[key] # Access metadata directly
        
        new_data_lists.append(groups)
    
    # Create a new dataset instance from the lists of new data
    print("Create a new dataset instance from the splitted time series...")
    dataset = Dataset.from_list(new_data_lists, features=dataset_obj.indexer.dataset.features)
    return to_timeseries_dataset(dataset)

def train(
        model_name=MODEL_NAME,
        model_path=MODEL_PATH,
        device_map=DEVICE_MAP,
        epochs=EPOCHS,
        patience=PATIENCE,
        data_from_splitted_files=True,
        test_size=TEST_SIZE,
        batch_size=2,
        max_sequence_length=None,#1024,
        min_patches=16,
        min_mask_ratio=0.2,
        max_mask_ratio=0.5,
        max_dim=1024,
        beta1=0.9,
        beta2=0.98,
        loss_func=PackedNLLLoss(),
        val_metric=PackedNLLLoss(),
        learning_rate=1e-3,
        weight_decay=1e-2
        ):
    print(f"Using device: {device_map}")

    # Model
    pretrained_module = MoiraiModule.from_pretrained(model_path).to(device_map)

    model = MoiraiFinetune(
        module=pretrained_module,
        min_patches=min_patches,
        min_mask_ratio=min_mask_ratio,
        max_mask_ratio=max_mask_ratio,
        max_dim=max_dim,
        num_training_steps=10000,
        num_warmup_steps=1000,
        module_kwargs=None,
        num_samples=100,
        beta1=beta1,
        beta2=beta2,
        loss_func=loss_func,
        val_metric=val_metric,
        lr=learning_rate,
        weight_decay=weight_decay,
        log_on_step=False,
    )

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")

    # Trainer
    logger = CSVLogger("logs", name=model_name)
    checkpoint_all = ModelCheckpoint(
        dirpath="checkpoints",
        filename=f"{model_name}_epoch_{{epoch}}",
        every_n_epochs=1,
        save_top_k=-1
    )
    checkpoint_best = ModelCheckpoint(
        dirpath="checkpoints",
        filename=f"{model_name}_best",
        monitor="val/PackedNLLLoss",
        mode="min",
        save_top_k=1
    )
    early_stopping = EarlyStopping(
        monitor="val/PackedNLLLoss",
        mode="min",
        patience=patience
    )

    logger = CSVLogger("logs", name=model_name)
    stats_logger = EpochStatsLogger(model_name)

    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
        max_epochs=epochs,
        logger=logger,
        callbacks=[checkpoint_all, checkpoint_best, early_stopping, stats_logger],
        log_every_n_steps=50,
        accumulate_grad_batches=1, # default
        #precision="16-mixed",
    )

    # Dataset
    if data_from_splitted_files and os.path.exists("data/train_dataset.pkl") and os.path.exists("data/val_dataset.pkl"):
        with open("data/train_dataset.pkl", "rb") as f:
            train_dataset = pickle.load(f)
        with open("data/val_dataset.pkl", "rb") as f:
            val_dataset = pickle.load(f)
    else:
        train_dataset, val_dataset = get_train_and_val_datasets(test_size=test_size)

    #train_dataset= split_long_time_series(train_dataset, max_seq_len=max_sequence_length) # TODO:
    #val_dataset= split_long_time_series(val_dataset, max_seq_len=max_sequence_length)
    lengths = np.asarray([len(sample) for sample in train_dataset.indexer.dataset.data["target"]])
    max_length = lengths.max() if lengths.size > 0 else 0
    lengths = np.asarray([len(sample) for sample in val_dataset.indexer.dataset.data["target"]])
    max_length = max(max_length, lengths.max() if lengths.size > 0 else 0)

    max_sequence_length = max_length if max_sequence_length is None else max_sequence_length

    collate_fn = CostumPadCollate(
        seq_fields=["target", "observed_mask", "time_id", "variate_id", "prediction_mask"],
        target_field="target",
        pad_func_map={
            "target": pad_tensor,
            "observed_mask": pad_bool_tensor,
            "time_id": pad_int_tensor,
            "variate_id": pad_int_tensor,
            "prediction_mask": pad_bool_tensor,
        },
        max_length=max_length,
        max_sequence_length=max_sequence_length,
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    print(train_dataloader)

    # Train
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_sequence_length", type=int, default=None)
    parser.add_argument("--min_patches", type=int, default=16)
    parser.add_argument("--min_mask_ratio", type=float, default=0.2)
    parser.add_argument("--max_mask_ratio", type=float, default=0.5)
    parser.add_argument("--max_dim", type=int, default=1024)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)

    args = parser.parse_args()

    train(
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        max_sequence_length=args.max_sequence_length,
        min_patches=args.min_patches,
        min_mask_ratio=args.min_mask_ratio,
        max_mask_ratio=args.max_mask_ratio,
        max_dim=args.max_dim,
        beta1=args.beta1,
        beta2=args.beta2,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
