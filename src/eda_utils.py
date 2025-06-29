import pickle

import pandas as pd
from datasets import Dataset


def main():
    indexed_dataset = Dataset.load_from_disk("data/moirai_dataset")

    df = pd.DataFrame(columns=[
        "dataset", "ts_len", "num_variates", "item_id", "start", "freq"
    ])

    dataset, item_id, strat, freq, ts_len, num_variates = [], [], [], [], [], []

    # save features
    for sample in indexed_dataset:
        dataset.append(sample["dataset"])
        item_id.append(sample["item_id"])
        strat.append(sample["start"])
        freq.append(sample["freq"])
        ts_len.append(len(sample["target"]))
        num_variates.append(len(sample["target"][0]))

    # compose df
    df["dataset"] = dataset
    df["ts_len"] = ts_len
    df["num_variates"] = num_variates
    df["item_id"] = item_id
    df["start"] = strat
    df["freq"] = freq

    df.to_csv("results/dataset_info.csv")

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
        ts_len.append(len(sample))
        num_variates.append(len(sample[0]))
        split.append("train")
    
    for sample in val_dataset.indexer.dataset.data["target"]:
        ts_len.append(len(sample))
        num_variates.append(len(sample[0]))
        split.append("val")
    
    # compose df
    df["ts_len"] = ts_len
    df["num_variates"] = num_variates
    df["split"] = split

    df.to_csv("results/dataset_train_val_info.csv")

    ##############################################################################

    print("Done :)")

if __name__ == "__main__":
    main()