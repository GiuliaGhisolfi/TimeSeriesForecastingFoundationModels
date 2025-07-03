import pickle

import pandas as pd
from datasets import Dataset


def main():
    """
    ds = Dataset.load_from_disk("data/moirai_dataset")

    def extract_info(example):
        target = example["target"]
        return {
            "ts_len": len(target),
            "num_variates": len(target[0]) if target else 0,
        }

    ds = ds.map(extract_info, desc="Extracting time series info")
    df = ds.to_pandas()[["dataset", "item_id", "start", "freq", "ts_len", "num_variates"]]
    df.to_csv("results/dataset_info.csv", index=False)

    ##############################################################################

    with open("data/train_dataset_full_ts.pkl", "rb") as f:
        train_dataset = pickle.load(f)
    
    with open("data/val_dataset_full_ts.pkl", "rb") as f:
        val_dataset = pickle.load(f)
    
    df = pd.DataFrame(columns=[
        "ts_len", "num_variates", "split",
    ])

    # ts lenghts
    ts_len, num_variates = [], []
    split = []

    for sample in train_dataset.indexer.dataset.data["target"]:
        num_variates.append(len(sample))
        ts_len.append(len(sample[0]))
        split.append("train")
    
    for sample in val_dataset.indexer.dataset.data["target"]:
        num_variates.append(len(sample[0]))
        ts_len.append(len(sample))
        split.append("val")
    
    # compose df
    df["num_variates"] = num_variates
    df["ts_len"] = ts_len
    df["split"] = split

    df.to_csv("results/dataset_train_val_info.csv")

    ##############################################################################
    """

    ds = Dataset.load_from_disk("data/moirai_dataset_splitted")

    def extract_info(example):
        target = example["target"]
        return {
                "ts_len": len(target),
                "num_variates": len(target[0]) if target else 0,
            }

    ds = ds.map(extract_info, desc="Extracting time series info")
    df = ds.to_pandas()[["dataset", "item_id", "start", "freq", "ts_len", "num_variates"]]
    df.to_csv("results/dataset_splitted_info.csv", index=False)

    ##############################################################################

    with open("data/train_dataset.pkl", "rb") as f:
        train_dataset = pickle.load(f)
    
    with open("data/val_dataset.pkl", "rb") as f:
        val_dataset = pickle.load(f)
    
    df = pd.DataFrame(columns=[
        "ts_len", "num_variates", "split",
    ])

    # ts lenghts
    ts_len, num_variates = [], []
    split = []

    for sample in train_dataset.indexer.dataset.data["target"]:
        num_variates.append(len(sample[0]))
        ts_len.append(len(sample))
        split.append("train")

    for sample in val_dataset.indexer.dataset.data["target"]:
        num_variates.append(len(sample[0]))
        ts_len.append(len(sample))
        split.append("val")

    # compose df
    df["num_variates"] = num_variates
    df["ts_len"] = ts_len
    df["split"] = split

    df.to_csv("results/dataset_splitted_train_val_info.csv")

    print("Done :)")

if __name__ == "__main__":
    main()