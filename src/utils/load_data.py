from datasets import load_dataset

CHRONOS = "autogluon/chronos_datasets"
DATASET_CHRONOS = [
    "dominick",
    "ercot",
    "exchange_rate",
    #"monash_m3_monthly"
]

def load_data(dataset_name):
    if dataset_name in DATASET_CHRONOS:
        return load_dataset(CHRONOS, dataset_name, trust_remote_code=True)