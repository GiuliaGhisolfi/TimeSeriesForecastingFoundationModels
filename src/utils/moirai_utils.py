import os
import random

import pandas as pd
from datasets import Dataset
from torch.utils.data._utils.collate import default_collate

from utils.load_moirai_data import load_data

RANDOM_SEED = 42
TEST_SIZE = 0.2

def stratified_split(dataset, stratify_col="dataset", test_size=TEST_SIZE, seed=RANDOM_SEED):
    groups = {}
    for row in dataset:
        key = row[stratify_col]
        groups.setdefault(key, []).append(row)

    train_splits, val_splits = [], []
    rng = random.Random(seed)

    for group_rows in groups.values():
        rng.shuffle(group_rows)
        n_test = int(len(group_rows) * test_size)
        val_splits.extend(group_rows[:n_test])
        train_splits.extend(group_rows[n_test:])

    train_dataset = Dataset.from_list(train_splits, features=dataset.features)
    val_dataset = Dataset.from_list(val_splits, features=dataset.features)

    return train_dataset, val_dataset

def get_train_and_val_datasets(yaml_path="data/datasets.yaml", stratify_col="dataset", test_size=TEST_SIZE, seed=RANDOM_SEED):
    # Check if datset are already loaded
    if os.path.exists("data/moirai/train_dataset") and os.path.exists("data/moirai/val_dataset"):
        print("Train and validation datasets already exist. Loading from disk...")

        train_dataset = Dataset.load_from_disk("data/moirai/train_dataset")
        val_dataset = Dataset.load_from_disk("data/moirai/val_dataset")
    
    else:
        print("Train and validation datasets do not exist. Loading from YAML and splitting...")

        # Load train and validation data
        full_dataset = load_data(yaml_path=yaml_path)
        train_dataset, val_dataset = stratified_split(
            full_dataset, stratify_col=stratify_col, test_size=test_size, seed=seed)
    
    return train_dataset, val_dataset

def save_train_and_val_datasets(yaml_path="data/datasets.yaml", stratify_col="dataset", test_size=TEST_SIZE, seed=RANDOM_SEED):
    full_dataset = load_data(yaml_path=yaml_path)
    train_dataset, val_dataset = stratified_split(
        full_dataset, stratify_col=stratify_col, test_size=test_size, seed=seed)

    # Save datasets to disk
    os.makedirs("data/moirai", exist_ok=True)
    train_path = "data/moirai/train_dataset"
    val_path = "data/moirai/val_dataset"

    train_dataset.save_to_disk(train_path)
    val_dataset.save_to_disk(val_path)

    print(f"Train dataset saved to {train_path}")
    print(f"Validation dataset saved to {val_path}")

def custom_collate_fn(batch):
    for i, item in enumerate(batch):
        for k in list(item.keys()):
            if isinstance(item[k], pd.Timestamp):
                # Convert to float (UNIX time) or string
                item[k] = item[k].timestamp()
    return default_collate(batch)
