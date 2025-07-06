import copy
import json
import os
import random

import pandas as pd
import torch
from datasets import Dataset, concatenate_datasets, load_from_disk
from tqdm import tqdm

from compose_moirai_dataset import concatenate_moirai_datasets
from moirai_utils.load_moirai_data import load_data
from moirai_utils.split_long_series_dataset import split_long_series_dataset
from moirai_utils.transformations import ToTorch
from uni2ts.data.dataset import SampleTimeSeriesType, TimeSeriesDataset
from uni2ts.data.indexer.hf_dataset_indexer import HuggingFaceDatasetIndexer

RANDOM_SEED = 42
TEST_SIZE = 0.2

def pad_tensor(shape, dtype=torch.float32):
    return torch.zeros(shape, dtype=dtype)

def pad_bool_tensor(shape, dtype=torch.bool):
    return torch.zeros(shape, dtype=dtype) # False

def pad_int_tensor(shape, pad_value=-1, dtype=torch.long):
    return torch.full(shape, pad_value, dtype=dtype)

def stratified_split(dataset, stratify_col="dataset", test_size=TEST_SIZE, seed=RANDOM_SEED):
    print(f"Stratified split with test size: {test_size}, seed: {seed}")

    os.makedirs("data/grouped", exist_ok=True)
    groups_files = {}

    for row in tqdm(dataset, desc="Writing per group"):
        key = row[stratify_col]
        safe_key = key.replace("/", "_")
        path = f"data/grouped/{safe_key}.jsonl"

        if key not in groups_files:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            groups_files[key] = open(path, "a")

        row_copy = copy.deepcopy(row)
        for k, v in row_copy.items():
            if isinstance(v, pd.Timestamp):
                row_copy[k] = v.isoformat()

        json.dump(row_copy, groups_files[key])
        groups_files[key].write("\n")

    for f in groups_files.values():
        f.close()

    os.makedirs("data/tmp_train", exist_ok=True)
    os.makedirs("data/tmp_val", exist_ok=True)

    train_idx = val_idx = 0
    train_chunk, val_chunk = [], []
    chunk_size = 100000

    rng = random.Random(seed)

    for filename in os.listdir("data/grouped"):
        with open(f"data/grouped/{filename}") as f:
            rows = [json.loads(line) for line in f]
        rng.shuffle(rows)
        n_val = int(len(rows) * test_size)
        val_chunk.extend(rows[:n_val])
        train_chunk.extend(rows[n_val:])

        if len(train_chunk) >= chunk_size:
            Dataset.from_list(train_chunk, features=dataset.features).save_to_disk(f"data/tmp_train/chunk_{train_idx}")
            train_chunk = []
            train_idx += 1

        if len(val_chunk) >= chunk_size:
            Dataset.from_list(val_chunk, features=dataset.features).save_to_disk(f"data/tmp_val/chunk_{val_idx}")
            val_chunk = []
            val_idx += 1

    if train_chunk:
        Dataset.from_list(train_chunk, features=dataset.features).save_to_disk(f"data/tmp_train/chunk_{train_idx}")
    if val_chunk:
        Dataset.from_list(val_chunk, features=dataset.features).save_to_disk(f"data/tmp_val/chunk_{val_idx}")

    # Concatenate all datas
    train_datasets = [load_from_disk(f"data/tmp_train/chunk_{i}") for i in range(train_idx + 1)]
    val_datasets = [load_from_disk(f"data/tmp_val/chunk_{i}") for i in range(val_idx + 1)]

    train_dataset = concatenate_datasets(train_datasets)
    val_dataset = concatenate_datasets(val_datasets)

    print(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")
    return train_dataset, val_dataset

def to_timeseries_dataset(
    indexed_dataset,
    context_length=2048,
    prediction_length=256,
    sample_time_series=SampleTimeSeriesType.NONE
):
    transform = ToTorch(context_length=context_length, prediction_length=prediction_length)
    indexer = HuggingFaceDatasetIndexer(indexed_dataset)
    return TimeSeriesDataset(
        indexer=indexer,
        transform=transform,
        sample_time_series=sample_time_series, # SampleTimeSeriesType.NONE/UNIFORM/PROPORTIONAL
    )

def get_train_and_val_datasets(stratify_col="dataset", context_length=2048, prediction_length=256,
        test_size=TEST_SIZE, seed=RANDOM_SEED):
    # Check if datset are already loaded
    if os.path.exists("data/final_split_dataset"):
        print("Loading from disk...")
        indexed_dataset = Dataset.load_from_disk("data/final_split_dataset")
    elif all(os.path.exists(f"data/split_part_{i}.arrow") for i in range(8)):
        print("Concatenate dataset...")
        num_chunks = 8
        all_chunks = [load_from_disk(f"data/split_part_{i}.arrow") for i in range(num_chunks)]
        indexed_dataset = concatenate_datasets(all_chunks)
    else:
        if os.path.exists("data/moirai_dataset"):
            print("Indexed datasets already exist. Loading from disk...")
            # Load train and validation data
            indexed_dataset = Dataset.load_from_disk("data/moirai_dataset")
        
        elif os.path.exists("data/splitted_moirai_dataset"):
            print("Splitted datasets already exist. Loading from disk...")
            indexed_dataset = concatenate_moirai_datasets()
            
        else:
            print("Train and validation datasets do not exist. Loading from YAML and splitting...")
            # Load the full dataset from YAML
            indexed_dataset = load_data(yaml_path="data/datasets.yaml")
            # and save it to disk
            save_train_and_val_datasets("data/datasets.yaml", "data/moirai_dataset")
        
        # Split time series
        indexed_dataset = split_long_series_dataset(
            indexed_dataset,
            context_length=context_length,
            prediction_length=prediction_length
            )

    # Stratified split
    train_dataset, val_dataset = stratified_split(
        indexed_dataset, stratify_col=stratify_col, test_size=test_size, seed=seed)

    # Convert to TimeSeriesDataset
    train_dataset = to_timeseries_dataset(
        train_dataset,
        context_length=context_length,
        prediction_length=prediction_length
        )
    val_dataset = to_timeseries_dataset(
        val_dataset,
        context_length=context_length,
        prediction_length=prediction_length
        )

    return train_dataset, val_dataset

def save_train_and_val_datasets(yaml_path="data/datasets.yaml", dataset_path="data/moirai_dataset"):
    full_dataset = load_data(yaml_path=yaml_path)

    # Save datasets to disk
    os.makedirs("data", exist_ok=True)
    full_dataset.save_to_disk(dataset_path)

    print(f"Dataset saved to {dataset_path}")

