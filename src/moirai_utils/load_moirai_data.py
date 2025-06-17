import os

import numpy as np
import pandas as pd
import yaml
from datasets import Dataset, Features, Sequence, Value, load_dataset

from uni2ts.common.typing import MultivarTimeSeries

RANDOM_SEED = 42
LOTSA_FRACTION = 0.05

FREQ_MAP_ETT = {
    "m1": "15T",
    "m2": "15T",
    "h1": "1H",
    "h2": "1H"
}

def infer_frequency_from_timestamps(timestamps):
    if len(timestamps) < 2:
        return "unknown"

    # Convert first two to pandas datetime
    t0 = pd.to_datetime(timestamps[0])
    t1 = pd.to_datetime(timestamps[1])

    delta = t1 - t0

    seconds = int(delta.total_seconds())
    
    if seconds < 60:
        return f"{seconds}S"
    elif seconds < 3600:
        return f"{seconds // 60}T"
    elif seconds < 86400:
        return f"{seconds // 3600}H"
    elif seconds < 604800:
        return f"{seconds // 86400}D"
    elif seconds < 2592000:
        return f"{seconds // 604800}W"
    else:
        return f"{seconds // 2592000}M"
    
def save_dataset_to_disk(dataset, path="data/splitted_moirai_dataset"):
    dataset_name = dataset["dataset"][0]
    save_path = f"{path}/{dataset_name.replace('/', '_')}"

    os.makedirs(save_path, exist_ok=True)
    dataset.save_to_disk(save_path)

    print(f"Dataset {dataset_name} saved to {save_path}")

def load_dataset_from_disk(dataset_name, path="data/splitted_moirai_dataset"):
    save_path = f"{path}/{dataset_name.replace('/', '_')}"
    
    if os.path.exists(save_path):
        return Dataset.load_from_disk(save_path)
    else:
        raise FileNotFoundError(f"Dataset {dataset_name} not found in {save_path}")

def unify_target_shape(example):
    if isinstance(example["target"][0], float):
        example["target"] = [[float(v)] for v in example["target"]]
    else:
        example["target"] = [[float(val) for val in row] for row in example["target"]]
    return example

def load_data(yaml_path="data/datasets.yaml"):
    datasets_list = []
    dataset_name_list = []

    # Load dataset names from YAML
    with open(yaml_path, "r") as f:
        datasets_config = yaml.safe_load(f)

    for group_name, dataset_names in datasets_config.items():
        print(f"\nLoading group: {group_name}")

        if group_name in ["autogluon/chronos_datasets", "autogluon/chronos_datasets_extra"]:
            for dataset_name in dataset_names:
                print(f"Loading {dataset_name}...")
                try:
                    ds = load_dataset(group_name, dataset_name, split="train", trust_remote_code=True)

                    # Adapt to LOTSA structure
                    if "id" in ds.column_names:
                        ds = ds.rename_column("id", "item_id")
                    if "target" not in ds.column_names:
                        target_name = ds.column_names[-1]
                        ds = ds.rename_column(target_name, "target")
                    # Add "start" and "freq"
                    data = [x for x in ds]
                    df = pd.DataFrame(data)
                    freq = df["timestamp"].apply(infer_frequency_from_timestamps)
                    ds = ds.add_column("freq", freq)
                    start = df["timestamp"].apply(lambda ts: ts[0] if isinstance(ts, list) and len(ts) > 0 else None)
                    start = pd.to_datetime(start, errors="coerce").astype("datetime64[ns]")
                    ds = ds.add_column("start", start)
                    # Remove unnecessary features
                    ds = ds.remove_columns([col for col in ds.column_names if col not in ["item_id", "start", "freq", "target"]])
                    # Dataset name
                    ds = ds.add_column("dataset", [group_name + "/" + dataset_name] * len(ds))

                    #datasets_list.append(ds)
                    save_dataset_to_disk(ds)
                    dataset_name_list.append(ds["dataset"][0].replace("/", "_"))

                except Exception as e:
                    print(f"Failed to load {dataset_name} from {group_name}: {e}")

        elif group_name == "ett":
            for dataset_name in dataset_names:
                print(f"Loading {dataset_name}...")
                try:
                    ds = load_dataset(group_name, dataset_name, split="train")

                    # Adapt to LOTSA structure
                    freq_value = FREQ_MAP_ETT.get(dataset_name)
                    ds = ds.add_column("freq", [freq_value] * len(ds))
                    # Cast "start" to timestamp[ns]
                    ds = ds.cast_column("start", Value("timestamp[ns]"))
                    # Remove unnecessary features
                    ds = ds.remove_columns([col for col in ds.column_names if col not in ["item_id", "start", "freq", "target"]])
                    # Dataset name
                    ds = ds.add_column("dataset", [group_name + "/" + dataset_name] * len(ds))

                    #datasets_list.append(ds)
                    save_dataset_to_disk(ds)
                    dataset_name_list.append(ds["dataset"][0].replace("/", "_"))

                except Exception as e:
                    print(f"Failed to load {dataset_name} from {group_name}: {e}")

        elif group_name == "Salesforce/lotsa_data":
            for dataset_name in dataset_names:
                print(f"Loading {dataset_name}...")
                try:
                    full_ds = load_dataset(group_name, dataset_name, split="train", trust_remote_code=True)
                    num_samples = int(LOTSA_FRACTION * len(full_ds))
                    ds = full_ds.shuffle(seed=RANDOM_SEED).select(range(num_samples))

                    # Cast "start" to timestamp[ns]
                    ds = ds.cast_column("start", Value("timestamp[ns]"))
                    # Remove unnecessary features
                    ds = ds.remove_columns([col for col in ds.column_names if col not in ["item_id", "start", "freq", "target"]])
                    # Dataset name
                    ds = ds.add_column("dataset", [group_name + "/" + dataset_name] * len(ds))

                    #datasets_list.append(ds)
                    save_dataset_to_disk(ds)
                    dataset_name_list.append(ds["dataset"][0].replace("/", "_"))

                except Exception as e:
                    print(f"Failed to load {dataset_name} from {group_name}: {e}")

        else:
            print(f"Unknown group: {group_name}, skipping...")
            continue

    # Concatenate all datasets
    if dataset_name_list:
        for ds_name in dataset_name_list:
            ds = load_dataset_from_disk(ds_name)
            datasets_list.append(ds)

        indexed_data = []
        for ds in datasets_list:
            ds = ds.map(unify_target_shape)
            for example in ds:
                ts_data: dict[str, MultivarTimeSeries] = {
                    "target": [np.array(dim, dtype=np.float32) for dim in example["target"]],
                    "item_id": example["item_id"],
                    "start": example["start"],
                    "freq": example["freq"],
                    "dataset": example["dataset"]
                }
                indexed_data.append(ts_data)

        features = Features({
            "item_id": Value("string"),
            "start": Value("timestamp[ns]"),
            "freq": Value("string"),
            "target": Sequence(Sequence(Value("float32"))),
            "dataset": Value("string"),
        })

        indexed_dataset = Dataset.from_list(indexed_data, features=features)

        return indexed_dataset

    else:
        raise ValueError("No datasets were loaded successfully.")