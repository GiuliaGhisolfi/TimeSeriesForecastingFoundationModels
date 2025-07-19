from torch.utils.data import Dataset, ConcatDataset
import os
from datasets import Dataset, concatenate_datasets, load_from_disk
from tqdm import tqdm
import numpy as np
from timesfm.finetuning.finetuning_example import prepare_datasets

#from timesfm_utils.dataset_list import dataset_name_list
dataset_name_list = [
    "autogluon_chronos_datasets_electricity_15min",
    "autogluon_chronos_datasets_mexico_city_bikes",]

DATA_PATH = "/raid/decaro/TimeSeriesForecastingFoundationModels/data/" # "data/"
TEST_SIZE = 0.2
RANDOM_STATE = 42

def load_dataset_from_disk(dataset_name, path=f"{DATA_PATH}splitted_moirai_dataset"):
    save_path = f"{path}/{dataset_name.replace('/', '_')}"
    
    if os.path.exists(save_path):
        return Dataset.load_from_disk(save_path)
    else:
        raise FileNotFoundError(f"Dataset {dataset_name} not found in {save_path}")
    
def to_numpy(row):
    row["target"] = np.array(row["target"])
    return row

def get_data(context_length: int, horizon_length: int) -> tuple[Dataset, Dataset]:
    train_data, val_data = [], []

    for ds_name in tqdm(dataset_name_list, desc="Preparing data"):
        ds = load_dataset_from_disk(ds_name)

        """n_val = int(len(ds) * TEST_SIZE)

        partial_train_dataset = Dataset.from_dict({
            "target": ds["target"][:-n_val],
            "item_id": ds["item_id"][:-n_val],
            "start": ds["start"][:-n_val],
            "freq": ds["freq"][:-n_val],
            "dataset": ds["dataset"][:-n_val]
            })
        
        partial_val_dataset = Dataset.from_dict({
            "target": ds["target"][-n_val:],
            "item_id": ds["item_id"][-n_val:],
            "start": ds["start"][-n_val:],
            "freq": ds["freq"][-n_val:],
            "dataset": ds["dataset"][-n_val:]
        })"""

        #series = np.array([np.array(s) for s in ds["target"]])
        ds = ds.map(to_numpy, num_proc=8)

        partial_train_dataset, partial_val_dataset = prepare_datasets(
            ds["target"],
            context_length=context_length,
            horizon_length=horizon_length,
            freq_type=0,
            train_split=1-TEST_SIZE,
        )

        train_data.append(partial_train_dataset)
        val_data.append(partial_val_dataset)

    train_dataset = ConcatDataset(train_data)
    val_dataset = ConcatDataset(val_data)

    print(f"Train dataset size: {len(train_dataset)}")

    return train_dataset, val_dataset

