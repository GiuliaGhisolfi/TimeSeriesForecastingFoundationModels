
from moirai_utils.moirai_utils import get_train_and_val_datasets

train_dataset, val_dataset = get_train_and_val_datasets(test_size=0.2,
    dataset_path="data/splitted_moirai_dataset/autogluon_chronos_datasets_mexico_city_bikes")