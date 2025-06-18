import os
import random

import torch
from datasets import Dataset

from compose_moirai_dataset import concatenate_moirai_datasets
from moirai_utils.load_moirai_data import load_data
from moirai_utils.transformations import ToTorch
from uni2ts.data.dataset import SampleTimeSeriesType, TimeSeriesDataset
from uni2ts.data.indexer.hf_dataset_indexer import HuggingFaceDatasetIndexer

RANDOM_SEED = 42
TEST_SIZE = 0.2

def pad_tensor(shape, dtype=torch.float32):
        return torch.zeros(shape, dtype=dtype)

def stratified_split(dataset, stratify_col="dataset", test_size=TEST_SIZE, seed=RANDOM_SEED):
    groups = {}
    for row in dataset:
        key = row[stratify_col]
        groups.setdefault(key, []).append(row)

    train_splits, val_splits = [], []
    rng = random.Random(seed)

    for group_rows in groups.values():
        rng.shuffle(group_rows)
        n_val = int(len(group_rows) * test_size)
        val_splits.extend(group_rows[:n_val])
        train_splits.extend(group_rows[n_val:])

    #return train_splits, val_splits
    train_dataset = Dataset.from_list(train_splits, features=dataset.features)
    val_dataset = Dataset.from_list(val_splits, features=dataset.features)

    return train_dataset, val_dataset

def to_timeseries_dataset(indexed_dataset, transform=ToTorch(), sample_time_series=SampleTimeSeriesType.PROPORTIONAL):
    indexer = HuggingFaceDatasetIndexer(indexed_dataset)
    return TimeSeriesDataset(
        indexer=indexer,
        transform=transform, # Identity == no transformation is needed
        sample_time_series=sample_time_series, # SampleTimeSeriesType.NONE/UNIFORM/PROPORTIONAL
    )

def get_train_and_val_datasets(dataset_path="data/moirai_dataset", yaml_path="data/datasets.yaml",
        stratify_col="dataset", test_size=TEST_SIZE, seed=RANDOM_SEED):
    # Check if datset are already loaded
    if os.path.exists("data/moirai_dataset"):
        print("Train and validation datasets already exist. Loading from disk...")
        # Load train and validation data
        indexed_dataset = Dataset.load_from_disk(dataset_path)
    elif os.path.exits("data\splitted_moirai_dataset"):
        print("Splitted datasets already exist. Loading from disk...")
        indexed_dataset = concatenate_moirai_datasets()
        
    else:
        print("Train and validation datasets do not exist. Loading from YAML and splitting...")
        # Load the full dataset from YAML
        indexed_dataset = load_data(yaml_path=yaml_path)
        # and save it to disk
        save_train_and_val_datasets(yaml_path, dataset_path)

    # Stratified split
    train_dataset, val_dataset = stratified_split(
        indexed_dataset, stratify_col=stratify_col, test_size=test_size, seed=seed)

    return to_timeseries_dataset(train_dataset), to_timeseries_dataset(val_dataset)

def save_train_and_val_datasets(yaml_path="data/datasets.yaml", dataset_path="data/moirai_dataset"):
    full_dataset = load_data(yaml_path=yaml_path)

    # Save datasets to disk
    os.makedirs("data", exist_ok=True)
    full_dataset.save_to_disk(dataset_path)

    print(f"Dataset saved to {dataset_path}")

