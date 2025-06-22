import os
import pickle
import time

import numpy as np
import orjson
import pyarrow.compute as pc
import torch
from torch.utils._pytree import tree_map

from moirai_utils.loader import CostumPadCollate
from moirai_utils.moirai_utils import get_train_and_val_datasets, pad_tensor
from uni2ts.data.loader import DataLoader, PackCollate, PadCollate
from uni2ts.loss.packed import PackedNLLLoss
from uni2ts.model.moirai import MoiraiFinetune, MoiraiModule

#from torch.utils.data import DataLoader as TorchDataLoader


MODEL_PATH = "Salesforce/moirai-1.0-R-small" # "Salesforce/moirai-1.0-R-base", "Salesforce/moirai-1.0-R-large"
MODEL_NAME = "moirai_small"
DEVICE_MAP = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 10
TEST_SIZE = 0.2
PATIENCE = 3 # Early stopping patience


def move_batch_to_device(batch, device):
    return tree_map(lambda x: x.to(device) if isinstance(x, torch.Tensor) else x, batch)

def train(
        model_name=MODEL_NAME,
        model_path=MODEL_PATH,
        device_map=DEVICE_MAP,
        # Training parameters
        epochs=EPOCHS,
        patience=PATIENCE,
        # Data parameters
        data_from_splitted_files=True,  # If True, use pre-split train/val datasets
        test_size=TEST_SIZE,
        batch_size=64,
        # Defaults for MoiraiFinetune
        min_patches=16,
        min_mask_ratio=0.2,
        max_mask_ratio=0.5,
        max_dim=1024,
        beta1=0.9,
        beta2=0.98,
        loss_func=PackedNLLLoss(),
        val_metric=PackedNLLLoss(),
        learning_rate=1e-3,
        weight_decay=1e-2
        ):
    print(f"Using device: {device_map}")

    # Load model from checkpoint
    pretrained_module = MoiraiModule.from_pretrained(model_path).to(device_map)

    model = MoiraiFinetune(
        module=pretrained_module,
        min_patches=min_patches, #16,
        min_mask_ratio=min_mask_ratio, #0.2,
        max_mask_ratio=max_mask_ratio, #0.5,
        max_dim=max_dim, #1024,
        num_training_steps=10000,
        num_warmup_steps=1000,
        module_kwargs=None,
        num_samples=100,
        beta1=beta1, #0.9,
        beta2=beta2, #0.98,
        loss_func=loss_func,
        val_metric=val_metric,
        lr=learning_rate,
        weight_decay=weight_decay,
        log_on_step=False,
    ).to(device_map)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    if isinstance(model, torch.nn.DataParallel):
        hparams = model.module._hparams
    else:
        hparams = model._hparams

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hparams['lr'],
        betas=(hparams['beta1'], hparams['beta2']),
        weight_decay=hparams['weight_decay']
        )

    # Load train and validation data
    if data_from_splitted_files and os.path.exists("data/train_dataset.pkl") and os.path.exists("data/val_dataset.pkl"):
        with open("data/train_dataset.pkl", "rb") as f:
            train_dataset = pickle.load(f)
        
        with open("data/val_dataset.pkl", "rb") as f:
            val_dataset = pickle.load(f)
    else:
        train_dataset, val_dataset = get_train_and_val_datasets(test_size=test_size)
    # TODO: REMOVE THIS
    ############################################################################################################
    """from datasets import Dataset, Features, Sequence, Value

    from moirai_utils.moirai_utils import stratified_split

    dataset = Dataset.load_from_disk("data/splitted_moirai_dataset/autogluon_chronos_datasets_mexico_city_bikes")

    def unify_target_shape(example):
        if isinstance(example["target"][0], float):
            example["target"] = [[float(v)] for v in example["target"]]
        else:
            example["target"] = [[float(val) for val in row] for row in example["target"]]
        return example
    dataset = dataset.map(unify_target_shape)

    indexed_data = []
    for example in dataset:
        ts_data = {
            "target": [np.array(dim, dtype=np.float32) for dim in example["target"]],
            "item_id": example["item_id"],
            "start": example["start"],
            "freq": example["freq"],
            "dataset": example["dataset"]
        }
        indexed_data.append(ts_data)

    features = Features({
        "item_id": Value("string"),
        "start": Value("timestamp[ns]"),
        "freq": Value("string"),
        "target": Sequence(Sequence(Value("float32"))),
        "dataset": Value("string"),
    })

    print("Creating indexed dataset...")
    indexed_dataset = Dataset.from_list(indexed_data, features=features)

    train_dataset, val_dataset = stratified_split(indexed_dataset)"""
    #########################################################################################

    # Find maximum sequence length for padding
    lengths = np.asarray([len(sample) for sample in train_dataset.indexer.dataset.data["target"]])
    max_length = lengths.max() if lengths.size > 0 else 0

    lengths = np.asarray([len(sample) for sample in val_dataset.indexer.dataset.data["target"]])
    max_length = max(max_length, lengths.max() if lengths.size > 0 else 0)

    #max_length = max(len(s["target"]) for s in train_dataset) # TODO: REMOVE THIS
    #max_length = max(max_length, max(len(s["target"]) for s in val_dataset)) # TODO: REMOVE THIS

    # Create collate function for padding sequences
    collate_fn = PadCollate(
        seq_fields=["target"],
        target_field="target",
        pad_func_map={"target": pad_tensor},
        max_length=max_length,
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    os.makedirs("checkpoints", exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0
    val_losses = []
    train_losses = []
    times = []

    model_to_use = model.module if hasattr(model, "module") else model

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        start_time = time.time()

        # Train
        model.train()
        train_loss_total = 0
        n_batches = 0

        for batch_idx, batch in enumerate(train_dataloader):
            batch = move_batch_to_device(batch, device_map)
            optimizer.zero_grad()
            loss = model_to_use.training_step(batch, batch_idx=batch_idx)
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item()
            n_batches += 1

        train_loss_avg = train_loss_total / n_batches
        train_losses.append(train_loss_avg)
        print(f"Epoch {epoch}: Train Loss = {train_loss_avg:.4f}")

        # Validation
        model.eval()
        val_loss_total = 0
        n_batches_val = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                batch = move_batch_to_device(batch, device_map)
                val_loss = model_to_use.validation_step(batch, batch_idx=batch_idx)
                val_loss_total += val_loss.item()
                n_batches_val += 1
        
        val_loss_avg = val_loss_total / n_batches_val
        val_losses.append(val_loss_avg)
        print(f"Epoch {epoch}: Val Loss = {val_loss_avg:.4f}")

        end_time = time.time()
        elapsed_time = end_time - start_time
        times.append(elapsed_time)
        print(f"Epoch {epoch + 1} completed in {elapsed_time:.2f} seconds.")

        # Early stopping
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            patience_counter = 0

            # Save best model
            if isinstance(model, torch.nn.DataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            torch.save({'state_dict': state_dict}, f"checkpoints/{model_name}_best.ckpt")
            print("Saved best model.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        # Save checkpoint per epoch
        if isinstance(model, torch.nn.DataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        torch.save({'state_dict': state_dict}, f"checkpoints/{model_name}_epoch_{epoch}.ckpt")

        print(f"Saved checkpoint for epoch {epoch}.")

    # Save validation and training losses
    os.makedirs("results", exist_ok=True)

    with open(f"results/{model_name}_train_losses.json", "w") as f:
        orjson.dump(train_losses, f)
    with open(f"results/{model_name}_val_losses.json", "w") as f:
        orjson.dump(val_losses, f)
    print("Saved training and validation losses.")
    with open(f"results/{model_name}_times.json", "w") as f:
        orjson.dump(times, f)
    print("Saved training times.")


if __name__ == "__main__":
    train()