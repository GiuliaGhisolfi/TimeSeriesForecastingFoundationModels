# save_datasets.py
import pickle

from moirai_utils.moirai_utils import (get_train_and_val_datasets,
                                       save_train_and_val_datasets)

if __name__ == "__main__":
    #save_train_and_val_datasets(yaml_path="data/datasets.yaml", dataset_path="data/moirai_dataset")

    # Stratified split
    train_dataset, val_dataset = get_train_and_val_datasets(test_size=0.2)

    # save the datasets to disk
    with open("data/train_dataset.pkl", "wb") as f:
        pickle.dump(train_dataset, f)
    
    with open("data/val_dataset.pkl", "wb") as f:
        pickle.dump(val_dataset, f)

    print("Train and validation datasets saved to disk.")