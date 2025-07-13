import os
import pickle
import random
from collections import defaultdict

import numpy as np
from datasets import Dataset, concatenate_datasets, load_from_disk
from tqdm import tqdm

from moirai_utils.split_long_series_dataset import split_long_series_dataset
from moirai_utils.transformations import ToTorch
from uni2ts.data.dataset import SampleTimeSeriesType, TimeSeriesDataset
from uni2ts.data.indexer.hf_dataset_indexer import HuggingFaceDatasetIndexer

from collections import defaultdict
import shutil

RANDOM_SEED = 42
TEST_SIZE = 0.2

DATA_PATH = "/raid/decaro/TimeSeriesForecastingFoundationModels/data/" # "data/"

dataset_name_list = [
    "autogluon_chronos_datasets_electricity_15min",
    "autogluon_chronos_datasets_mexico_city_bikes",
    "autogluon_chronos_datasets_monash_electricity_hourly",
    "autogluon_chronos_datasets_monash_electricity_weekly",
    "autogluon_chronos_datasets_taxi_1h",
    "ett_h1",
    "ett_h2",
    "ett_m1",
    "ett_m2",
    "Salesforce_lotsa_data_alibaba_cluster_trace_2018",
    "Salesforce_lotsa_data_azure_vm_traces_2017",
    "Salesforce_lotsa_data_bdg-2_bear",
    "Salesforce_lotsa_data_bdg-2_fox",
    "Salesforce_lotsa_data_bdg-2_panther",
    "Salesforce_lotsa_data_bdg-2_rat",
    "Salesforce_lotsa_data_BEIJING_SUBWAY_30MIN",
    "Salesforce_lotsa_data_borg_cluster_data_2011",
    "Salesforce_lotsa_data_cmip6_1850",
    "Salesforce_lotsa_data_cmip6_1855",
    "Salesforce_lotsa_data_cmip6_1860",
    "Salesforce_lotsa_data_cmip6_1865",
    "Salesforce_lotsa_data_cmip6_1870",
    "Salesforce_lotsa_data_cmip6_1875",
    "Salesforce_lotsa_data_cmip6_1880",
    "Salesforce_lotsa_data_cmip6_1885",
    "Salesforce_lotsa_data_cmip6_1890",
    "Salesforce_lotsa_data_cmip6_1895",
    "Salesforce_lotsa_data_cmip6_1900",
    "Salesforce_lotsa_data_cmip6_1905",
    "Salesforce_lotsa_data_cmip6_1910",
    "Salesforce_lotsa_data_cmip6_1915",
    "Salesforce_lotsa_data_cmip6_1920",
    "Salesforce_lotsa_data_cmip6_1925",
    "Salesforce_lotsa_data_cmip6_1930",
    "Salesforce_lotsa_data_cmip6_1935",
    "Salesforce_lotsa_data_covid_deaths",
    "Salesforce_lotsa_data_covid_mobility",
    "Salesforce_lotsa_data_era5_1989",
    "Salesforce_lotsa_data_era5_1990",
    "Salesforce_lotsa_data_era5_1991",
    "Salesforce_lotsa_data_era5_1992",
    "Salesforce_lotsa_data_era5_1993",
    "Salesforce_lotsa_data_era5_1994",
    "Salesforce_lotsa_data_era5_1995",
    "Salesforce_lotsa_data_era5_1996",
    "Salesforce_lotsa_data_era5_1997",
    "Salesforce_lotsa_data_era5_1998",
    "Salesforce_lotsa_data_era5_1999",
    "Salesforce_lotsa_data_era5_2000",
    "Salesforce_lotsa_data_era5_2001",
    "Salesforce_lotsa_data_era5_2002",
    "Salesforce_lotsa_data_era5_2003",
    "Salesforce_lotsa_data_era5_2004",
    "Salesforce_lotsa_data_extended_web_traffic_with_missing",
    "Salesforce_lotsa_data_favorita_sales",
    "Salesforce_lotsa_data_favorita_transactions",
    "Salesforce_lotsa_data_fred_md",
    "Salesforce_lotsa_data_gfc12_load",
    "Salesforce_lotsa_data_godaddy",
    "Salesforce_lotsa_data_hierarchical_sales",
    "Salesforce_lotsa_data_hog",
    "Salesforce_lotsa_data_hospital",
    "Salesforce_lotsa_data_HZMETRO",
    "Salesforce_lotsa_data_ideal",
    "Salesforce_lotsa_data_kaggle_web_traffic_weekly",
    "Salesforce_lotsa_data_kdd_cup_2018_with_missing",
    "Salesforce_lotsa_data_kdd2022",
    "Salesforce_lotsa_data_largest",
    "Salesforce_lotsa_data_largest_2017",
    "Salesforce_lotsa_data_largest_2018",
    "Salesforce_lotsa_data_largest_2019",
    "Salesforce_lotsa_data_largest_2020",
    "Salesforce_lotsa_data_largest_2021",
    "Salesforce_lotsa_data_lcl",
    "Salesforce_lotsa_data_london_smart_meters_with_missing",
    "Salesforce_lotsa_data_LOOP_SEATTLE",
    "Salesforce_lotsa_data_LOS_LOOP",
    "Salesforce_lotsa_data_M_DENSE",
    "Salesforce_lotsa_data_m1_monthly",
    "Salesforce_lotsa_data_m1_quarterly",
    "Salesforce_lotsa_data_m1_yearly",
    "Salesforce_lotsa_data_m4_daily",
    "Salesforce_lotsa_data_m4_hourly",
    "Salesforce_lotsa_data_m4_monthly",
    "Salesforce_lotsa_data_m4_quarterly",
    "Salesforce_lotsa_data_m4_weekly",
    "Salesforce_lotsa_data_m4_yearly",
    "Salesforce_lotsa_data_m5",
    "Salesforce_lotsa_data_monash_m3_monthly",
    "Salesforce_lotsa_data_monash_m3_other",
    "Salesforce_lotsa_data_monash_m3_quarterly",
    "Salesforce_lotsa_data_monash_m3_yearly",
    "Salesforce_lotsa_data_nn5_daily_with_missing",
    "Salesforce_lotsa_data_nn5_weekly",
    "Salesforce_lotsa_data_pedestrian_counts",
    "Salesforce_lotsa_data_PEMS_BAY",
    "Salesforce_lotsa_data_PEMS03",
    "Salesforce_lotsa_data_PEMS04",
    "Salesforce_lotsa_data_PEMS07",
    "Salesforce_lotsa_data_PEMS08",
    "Salesforce_lotsa_data_project_tycho",
    "Salesforce_lotsa_data_Q-TRAFFIC",
    "Salesforce_lotsa_data_residential_load_power",
    "Salesforce_lotsa_data_residential_pv_power",
    "Salesforce_lotsa_data_restaurant",
    "Salesforce_lotsa_data_rideshare_with_missing",
    "Salesforce_lotsa_data_SHMETRO",
    "Salesforce_lotsa_data_subseasonal",
    "Salesforce_lotsa_data_subseasonal_precip",
    "Salesforce_lotsa_data_SZ_TAXI",
    "Salesforce_lotsa_data_taxi_30min",
    "Salesforce_lotsa_data_temperature_rain_with_missing",
    "Salesforce_lotsa_data_tourism_monthly",
    "Salesforce_lotsa_data_tourism_quarterly",
    "Salesforce_lotsa_data_tourism_yearly",
    "Salesforce_lotsa_data_traffic_hourly",
    "Salesforce_lotsa_data_traffic_weekly",
    "Salesforce_lotsa_data_uber_tlc_daily",
    "Salesforce_lotsa_data_uber_tlc_hourly",
    "Salesforce_lotsa_data_vehicle_trips_with_missing",
    "Salesforce_lotsa_data_weather",
    "Salesforce_lotsa_data_wiki-rolling_nips",
    "Salesforce_lotsa_data_wind_farms_with_missing"
]


def load_dataset_from_disk(dataset_name, path=f"{DATA_PATH}splitted_moirai_dataset"):
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

def stratified_split(test_size=TEST_SIZE, seed=RANDOM_SEED, chunk_size=5000):
    shutil.rmtree(f"{DATA_PATH}moirai_tmp", ignore_errors=True)
    os.makedirs(f"{DATA_PATH}moirai_tmp/train", exist_ok=True)
    os.makedirs(f"{DATA_PATH}moirai_tmp/val", exist_ok=True)

    rng = random.Random(seed)
    train_idx = val_idx = 0
    train_chunk = []
    val_chunk = []

    for ds_name in tqdm(dataset_name_list, desc="Loading + Splitting"):
        ds = load_dataset_from_disk(ds_name.replace("/", "_"), path=f"{DATA_PATH}moirai_dataset_processed")

        groups = defaultdict(list)
        for example in ds:
            key = example["dataset"]
            groups[key].append(example)

        for key, rows in groups.items():
            rng.shuffle(rows)
            n_val = int(len(rows) * test_size)
            val_rows = rows[:n_val]
            train_rows = rows[n_val:]

            train_chunk.extend(train_rows)
            val_chunk.extend(val_rows)

            if len(train_chunk) >= chunk_size:
                Dataset.from_list(train_chunk).save_to_disk(f"{DATA_PATH}moirai_tmp/train/chunk_{train_idx}")
                train_chunk = []
                train_idx += 1

            if len(val_chunk) >= chunk_size:
                Dataset.from_list(val_chunk).save_to_disk(f"{DATA_PATH}moirai_tmp/val/chunk_{val_idx}")
                val_chunk = []
                val_idx += 1
        
        if train_chunk:
            Dataset.from_list(train_chunk).save_to_disk(f"{DATA_PATH}moirai_tmp/train/chunk_{train_idx}")
            train_chunk = []
            train_idx += 1
        if val_chunk:
            Dataset.from_list(val_chunk).save_to_disk(f"{DATA_PATH}moirai_tmp/val/chunk_{val_idx}")
            val_chunk = []
            val_idx += 1

        del ds
        del groups

        print(f"{ds_name} done")

    if train_chunk:
        Dataset.from_list(train_chunk).save_to_disk(f"{DATA_PATH}moirai_tmp/train/chunk_{train_idx}")
    if val_chunk:
        Dataset.from_list(val_chunk).save_to_disk(f"{DATA_PATH}moirai_tmp/val/chunk_{val_idx}")

    train_datasets = [load_from_disk(f"{DATA_PATH}moirai_tmp/train/chunk_{i}") for i in range(train_idx + 1)]
    val_datasets = [load_from_disk(f"{DATA_PATH}moirai_tmp/val/chunk_{i}") for i in range(val_idx + 1)]

    train_dataset = concatenate_datasets(train_datasets)
    val_dataset = concatenate_datasets(val_datasets)

    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
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

def create_moirai_datasets(context_length=2048, prediction_length=256):
    os.makedirs(f"{DATA_PATH}moirai_dataset_processed", exist_ok=True)

    for ds_name in tqdm(dataset_name_list, desc="Preparing data"):
        ds = load_dataset_from_disk(ds_name)
        print(f"Loaded {ds_name} dataset.")
        ts_data_list = []

        ds = ds.map(unify_target_shape)
        for example in ds:
            ts_data = {
                "target": [np.array(dim, dtype=np.float32) for dim in example["target"]],
                "item_id": example["item_id"],
                "start": example["start"],
                "freq": example["freq"],
                "dataset": example["dataset"]
            }
            ts_data_list.append(ts_data)

        # Split time series
        indexed_dataset = split_long_series_dataset(
            ts_data_list,
            context_length=context_length,
            prediction_length=prediction_length
        )

        save_path = f"{ds_name.replace('/', '_')}"
        indexed_dataset.save_to_disk(f"{DATA_PATH}moirai_dataset_processed/{save_path}")

    print("Done :)")


def save_moirai_datasets(stratify_col="dataset", context_length=2048, prediction_length=256,
        test_size=TEST_SIZE, seed=RANDOM_SEED):
    # Save datasets to disk
    #os.makedirs(F"{DATA_PATH}moirai_dataset_processed", exist_ok=True)
    #create_moirai_datasets(context_length, prediction_length)

    # Stratified split
    train_dataset, val_dataset = stratified_split(
        test_size=test_size, seed=seed)

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

if __name__ == "__main__":
    train_dataset, val_dataset = save_moirai_datasets()

    # save the datasets to disk
    with open(f"{DATA_PATH}train_dataset.pkl", "wb") as f:
        pickle.dump(train_dataset, f)

    with open(f"{DATA_PATH}val_dataset.pkl", "wb") as f:
        pickle.dump(val_dataset, f)

    print("Train and validation datasets saved to disk.")