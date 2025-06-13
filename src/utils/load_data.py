from random import sample, seed

import pandas as pd
import yaml
from datasets import (Dataset, Value, concatenate_datasets,
                      interleave_datasets, load_dataset)

RANDOM_SEED = 42
LOTSA_FRACTION = 0.1

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

def load_data(yaml_path="data/datasets.yaml"):
    datasets_list = []

    # Load dataset names from YAML
    with open(yaml_path, "r") as f:
        datasets_config = yaml.safe_load(f)

    for group_name, dataset_names in datasets_config.items():
        print(f"\nLoading group: {group_name}")

        if group_name in ["autogluon/chronos_datasets", "autogluon/chronos_datasets_extra"]:
            for dataset_name in dataset_names[:1]:
                break #TODO
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

                    datasets_list.append(ds)
                except Exception as e:
                    print(f"Failed to load {dataset_name} from {group_name}: {e}")

        elif group_name == "ett":
            for dataset_name in dataset_names:
                print(f"Loading {dataset_name}...")
                try:
                    ds = load_dataset(group_name, dataset_name, trust_remote_code=True)["train"]

                    # Adapt to LOTSA structure
                    freq_value = FREQ_MAP_ETT.get(dataset_name)
                    ds = ds.add_column("freq", [freq_value] * len(ds))
                    # Cast "start" to timestamp[ns]
                    ds = ds.cast_column("start", Value("timestamp[ns]"))
                    # Remove unnecessary features
                    ds = ds.remove_columns([col for col in ds.column_names if col not in ["item_id", "start", "freq", "target"]])
                    # Dataset name
                    ds = ds.add_column("dataset", [group_name + "/" + dataset_name] * len(ds))

                    datasets_list.append(ds)
                except Exception as e:
                    print(f"Failed to load {dataset_name} from {group_name}: {e}")

        elif group_name == "Salesforce/lotsa_data":
            for dataset_name in dataset_names[1:2]:
                print(f"Loading {dataset_name}...")
                try:
                    full_ds = load_dataset(group_name, dataset_name, trust_remote_code=True)["train"]
                    num_samples = int(LOTSA_FRACTION * len(full_ds))
                    ds = full_ds.shuffle(seed=RANDOM_SEED).select(range(num_samples))

                    # Cast "start" to timestamp[ns]
                    ds = ds.cast_column("start", Value("timestamp[ns]"))
                    # Remove unnecessary features
                    ds = ds.remove_columns([col for col in ds.column_names if col not in ["item_id", "start", "freq", "target"]])
                    # Dataset name
                    ds = ds.add_column("dataset", [group_name + "/" + dataset_name] * len(ds))

                    datasets_list.append(ds)
                except Exception as e:
                    print(f"Failed to load {dataset_name} from {group_name}: {e}")

        else:
            print(f"Unknown group: {group_name}, skipping...")
            continue

    # Concatenate all datasets
    if datasets_list:
        from datasets import Dataset

        full_train_dataset = Dataset.from_dict({k: v for ds in datasets_list for k, v in ds.to_dict().items()})
        return full_train_dataset
    else:
        raise ValueError("No datasets were loaded successfully.")
    
if __name__ == "__main__":
    load_data()