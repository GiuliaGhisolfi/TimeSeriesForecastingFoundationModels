import torch.multiprocessing as mp
from torch.utils.data import Dataset

from timesfm.finetuning import get_model, setup_process
from timesfm.finetuning.finetuning_torch import FinetuningConfig


def get_data(batch_size: int, horizon_len: int) -> tuple[Dataset, Dataset]:
    pass

def train():
    gpu_ids = [] #TODO
    world_size = len(gpu_ids)

    model, hparams, tfm_config = get_model(load_weights=True)

    # Create config
    config = FinetuningConfig(
        batch_size=256,
        num_epochs=5,
        learning_rate=3e-5,
        use_wandb=True,
        distributed=True,
        gpu_ids=gpu_ids,
        log_every_n_steps=50,
        val_check_interval=0.5,
    )

    train_dataset, val_dataset = get_data(128, tfm_config.horizon_len)
    manager = mp.Manager()
    return_dict = manager.dict()

    # Launch processes
    mp.spawn(
        setup_process,
        args=(world_size, model, config, train_dataset, val_dataset, return_dict),
        nprocs=world_size,
        join=True,
    )

    results = return_dict.get("results", None)