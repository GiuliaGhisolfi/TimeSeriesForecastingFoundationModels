# save_datasets.py
from utils.moirai_utils import save_train_and_val_datasets

if __name__ == "__main__":
    save_train_and_val_datasets(yaml_path="data/datasets.yaml")