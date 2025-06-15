from random import sample, seed

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset, Features, Sequence, Value, load_dataset

from uni2ts.common.typing import MultivarTimeSeries, UnivarTimeSeries
from uni2ts.data.dataset import SampleTimeSeriesType, TimeSeriesDataset
from uni2ts.data.indexer.hf_dataset_indexer import HuggingFaceDatasetIndexer
from uni2ts.transform import Identity, Transformation

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

def unify_target_shape(example): #FIXME: REMOUVE THIS FUNCTION
    if isinstance(example["target"][0], float):
        example["target"] = [[float(v)] for v in example["target"]]
    else:
        example["target"] = [[float(val) for val in row] for row in example["target"]]
    return example

def convert_to_tensor(sample): # TODO: delete this function
    sample["target"] = torch.tensor(sample["target"], dtype=torch.float32)
    return sample

def ensure_target_format(example):
    current_target = example["target"]

    if not isinstance(current_target, list):
        current_target = current_target.tolist() if hasattr(current_target, 'tolist') else list(current_target)

    if not current_target or (isinstance(current_target[0], list) and all(isinstance(val, (int, float)) for val in current_target[0])):
        example["target"] = [[float(v) for v in inner_list] for inner_list in current_target]
    else:
        if all(isinstance(val, (int, float)) for val in current_target):
            example["target"] = [[float(v)] for v in current_target]
        else:
            raise ValueError(f"Unexpected target format for item_id {example.get('item_id', 'unknown')}: {current_target}")
    return example

def load_data(yaml_path="data/datasets.yaml"):
    datasets_list = []

    # Load dataset names from YAML
    with open(yaml_path, "r") as f:
        datasets_config = yaml.safe_load(f)

    for group_name, dataset_names in datasets_config.items():
        print(f"\nLoading group: {group_name}")

        if group_name in ["autogluon/chronos_datasets", "autogluon/chronos_datasets_extra"]:
            for dataset_name in dataset_names:
                break
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
                    #ds = ds.select_columns(['item_id', 'start', 'freq', 'target', 'dataset'])

                    datasets_list.append(ds)
                except Exception as e:
                    print(f"Failed to load {dataset_name} from {group_name}: {e}")

        elif group_name == "ett":
            for dataset_name in dataset_names:
                break
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
                    #ds = ds.select_columns(['item_id', 'start', 'freq', 'target', 'dataset'])

                    datasets_list.append(ds)
                except Exception as e:
                    print(f"Failed to load {dataset_name} from {group_name}: {e}")

        elif group_name == "Salesforce/lotsa_data":
            for dataset_name in dataset_names[2:4]:
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
                    #ds = ds.select_columns(['item_id', 'start', 'freq', 'target', 'dataset'])

                    datasets_list.append(ds)
                except Exception as e:
                    print(f"Failed to load {dataset_name} from {group_name}: {e}")

        else:
            print(f"Unknown group: {group_name}, skipping...")
            continue

    # Concatenate all datasets
    if datasets_list:
        unified_examples = []
        for ds in datasets_list:
            ds = ds.map(unify_target_shape)
            unified_examples.extend(ds)

        features = Features({
            "item_id": Value("string"),
            "start": Value("timestamp[ns]"),
            "freq": Value("string"),
            "target": Sequence(Sequence(Value("float32"))),
            "dataset": Value("string"),
        })
        full_train_dataset = Dataset.from_list(unified_examples, features=features)

        ############################# new code ##################################
        indexed_data = []
        for example in full_train_dataset:
            ts_data: dict[str, MultivarTimeSeries] = {
                "target": [np.array(dim, dtype=np.float32) for dim in example["target"]],
                "item_id": example["item_id"],
                "start": example["start"],
                "freq": example["freq"],
                "dataset": example["dataset"]
            }
            indexed_data.append(ts_data)

        indexed_data_hf = Dataset.from_list(indexed_data)
        indexer = HuggingFaceDatasetIndexer(indexed_data_hf)
        #indexer = HuggingFaceDatasetIndexer(indexed_data)

        # TimeSeriesDataset
        unified_dataset = TimeSeriesDataset(
            indexer=indexer,
            transform=Transformation(), # or Identity() if no transformation is needed
            sample_time_series=SampleTimeSeriesType.NONE, # NONE/UNIFORM/PROPORTIONAL
        )

        return unified_dataset
        #########################################################################

    else:
        raise ValueError("No datasets were loaded successfully.")