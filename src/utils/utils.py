import random

from datasets import Dataset

from src.utils.load_data import load_data

RANDOM_SEED = 42

def stratified_split(dataset, stratify_col="dataset", test_size=0.2, seed=RANDOM_SEED):
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

    # Ricostruisci come Dataset
    train_dataset = Dataset.from_list(train_splits, features=dataset.features)
    val_dataset = Dataset.from_list(val_splits, features=dataset.features)

    return train_dataset, val_dataset

def get_train_and_val_datasets(yaml_path="data/datasets.yaml", stratify_col="dataset", test_size=0.2, seed=RANDOM_SEED):
    # Check if datset are already loaded
    #TODO
    # Load train and validation data
    full_dataset = load_data(yaml_path=yaml_path)
    train_dataset, val_dataset = stratified_split(
        full_dataset, stratify_col=stratify_col, test_size=test_size, seed=seed)
    
    return train_dataset, val_dataset
