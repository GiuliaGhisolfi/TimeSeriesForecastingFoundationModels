import torch.multiprocessing as mp
import torch
import os
from timesfm_utils.load_data import get_data
from timesfm.finetuning.finetuning_example import setup_process
from timesfm.finetuning.finetuning_torch import FinetuningConfig

from timesfm import TimesFm, TimesFmCheckpoint, TimesFmHparams
from timesfm.pytorch_patched_decoder import PatchedTimeSeriesDecoder

DEVICE_MAP = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "google/timesfm-2.0-500m-pytorch"

EPOCHS = 2
PATIENCE = 3


def train(
    device=DEVICE_MAP,
    epochs=EPOCHS,
    batch_size=4,
    learning_rate=3e-5,
    context_len=2048, 
    horizon_len=256
    ):
    gpu_ids = [int(x) for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.isdigit()]
    world_size = len(gpu_ids)

    mp.set_start_method("spawn", force=True)
    
    # Load model
    hparams = TimesFmHparams(
        backend=device,
        per_core_batch_size=32,
        horizon_len=horizon_len,
        num_layers=50,
        use_positional_embedding=False,
        context_len=context_len,
    )
    tfm = TimesFm(
        hparams=hparams,
        checkpoint=TimesFmCheckpoint(huggingface_repo_id=MODEL_PATH))

    model = PatchedTimeSeriesDecoder(tfm._model_config)

    # Create config
    config = FinetuningConfig(
        batch_size=batch_size,
        num_epochs=epochs,
        learning_rate=learning_rate,
        use_wandb=False,
        distributed=True,
        gpu_ids=gpu_ids,
        log_every_n_steps=1,
        val_check_interval=0.5,
    )

    train_dataset, val_dataset = get_data(context_len, horizon_len)
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

    print(results)

if __name__ == "__main__":
    train(
        device=DEVICE_MAP,
        epochs=EPOCHS,
        batch_size=256,
        learning_rate=3e-5,
        context_len=2048, 
        horizon_len=128
    )