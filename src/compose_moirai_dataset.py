import os

import numpy as np
from datasets import Dataset, Features, Sequence, Value

from uni2ts.common.typing import MultivarTimeSeries

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


def concatenate_moirai_datasets():
    datasets_list = []

    # Concatenate all datasets
    for ds_name in dataset_name_list:
        ds = load_dataset_from_disk(ds_name)
        datasets_list.append(ds)
    print(f"Loaded {len(datasets_list)} datasets.")

    indexed_data = []
    for i, ds in enumerate(datasets_list):
        print(f"Processing dataset {i + 1}/{len(datasets_list)}")
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

    print("Creating indexed dataset...")
    indexed_dataset = Dataset.from_list(indexed_data, features=features)
    return indexed_dataset

def save_concatenate_moirai_datasets():
    indexed_dataset = concatenate_moirai_datasets()

    # Save datasets to disk
    dataset_path="data/moirai_dataset"

    os.makedirs("data", exist_ok=True)
    indexed_dataset.save_to_disk(dataset_path)

    print(f"Dataset saved to {dataset_path}")

if __name__ == "__main__":
    save_concatenate_moirai_datasets()
    print("Moirai datasets concatenated and saved successfully.")