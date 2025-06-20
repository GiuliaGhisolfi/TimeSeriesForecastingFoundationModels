# save_datasets.py
from moirai_utils.moirai_utils import (get_train_and_val_datasets,
                                       save_train_and_val_datasets)

if __name__ == "__main__":
    #save_train_and_val_datasets(yaml_path="data/datasets.yaml", dataset_path="data/moirai_dataset")

    # Stratified split
    train_dataset, val_dataset = get_train_and_val_datasets(test_size=0.2)

    # save the datasets to disk
    train_dataset.save_to_disk("data/train_dataset")
    val_dataset.save_to_disk("data/val_dataset")

    print("Train and validation datasets saved to disk.")