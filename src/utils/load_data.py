from datasets import load_dataset

CHRONOS = "autogluon/chronos_datasets"
DATASET_CHRONOS = [
    "electricity_15min",
    #"m4_daily",
    #"m4_hourly",
    #"m4_monthly",
    #"m4_weekly",
    "mexico_city_bikes",
    "monash_electricity_hourly",
    "monash_electricity_weekly",
    #"monash_kdd_cup_2018",
    #"monash_london_smart_meters",
    #"monash_pedestrian_counts",
    #"monash_rideshare",
    #"monash_saugeenday",
    #"monash_temperature_rain",
    #"monash_tourism_monthly",
    #"solar",
    #"solar_1h",
    "taxi_1h",
    #"taxi_30min",
    "training_corpus_kernel_synth_1m", # 10M TSMixup augmentations of real-world data
    "training_corpus_tsmixup_10m", # 1M synthetic time series generated with KernelSynth
    #"uber_tlc_daily",
    #"uber_tlc_hourly",
    "ushcn_daily",
    "weatherbench_daily",
    "weatherbench_hourly_10m_u_component_of_wind",
    "weatherbench_hourly_10m_v_component_of_wind",
    "weatherbench_hourly_2m_temperature",
    "weatherbench_hourly_geopotential",
    "weatherbench_hourly_potential_vorticity",
    "weatherbench_hourly_relative_humidity",
    "weatherbench_hourly_specific_humidity",
    "weatherbench_hourly_temperature",
    "weatherbench_hourly_toa_incident_solar_radiation",
    "weatherbench_hourly_total_cloud_cover",
    "weatherbench_hourly_total_precipitation",
    "weatherbench_hourly_u_component_of_wind",
    "weatherbench_hourly_v_component_of_wind",
    "weatherbench_hourly_vorticity",
    "weatherbench_weekly",
    "wiki_daily_100k",
    #"wind_farms_daily",
    #"wind_farms_hourly",
]

CHRONOS_EXTRA = "autogluon/chronos_datasets_extra"
DATASET_CHRONOS_EXTRA = [
    "spanish_energy_and_weather",
    "brazilian_cities_temperature",
]

ETT = "ett"
DATASET_ETT = [
    "m1",
    "m2",
    "h1",
    "h2",
]

def load_data():
    datasets_list = []

    # Load datasets from CHRONOS
    for dataset_name in DATASET_CHRONOS:
        print(f"Loading {dataset_name}...")
        try:
            ds = load_dataset(CHRONOS, dataset_name, split="train", trust_remote_code=True)
            datasets_list.append(ds)
        except Exception as e:
            print(f"Failed to load {dataset_name}: {e}")

    # Load datasets from CHRONOS_EXTRA
    for dataset_name in DATASET_CHRONOS_EXTRA:
        print(f"Loading {dataset_name}...")
        try:
            ds = load_dataset(CHRONOS_EXTRA, dataset_name, split="train", trust_remote_code=True)
            datasets_list.append(ds)
        except Exception as e:
            print(f"Failed to load {dataset_name}: {e}")

    # Concatenate all datasets
    if datasets_list:
        full_train_dataset = concatenate_datasets(datasets_list)
        return full_train_dataset
    else:
        raise ValueError("No datasets were loaded successfully.")