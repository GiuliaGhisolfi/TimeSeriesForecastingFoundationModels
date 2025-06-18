import json
import os
import time

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

def train():
    print(f"Using device: {DEVICE_MAP}")

    # Load model from checkpoint
    pretrained_module = MoiraiModule.from_pretrained(MODEL_PATH).to(DEVICE_MAP)

    model = MoiraiFinetune(
        module=pretrained_module,
        min_patches=16,
        min_mask_ratio=0.2,
        max_mask_ratio=0.5,
        max_dim=1024,
        num_training_steps=10000,
        num_warmup_steps=1000,
        module_kwargs=None,
        num_samples=100,
        beta1=0.9,
        beta2=0.98,
        loss_func=PackedNLLLoss(),
        val_metric=None,
        lr=1e-3,
        weight_decay=1e-2,
        log_on_step=False,
    ).to(DEVICE_MAP)

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
    train_dataset, val_dataset = get_train_and_val_datasets(test_size=TEST_SIZE)

    max_length = max(len(s["target"]) for s in train_dataset)
    max_length = max(max_length, max(len(s["target"]) for s in val_dataset))

    collate_fn = PadCollate( #or PackCollate
        seq_fields=["target"],
        target_field="target",
        pad_func_map={"target": pad_tensor},
        max_length=max_length,
    )

    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    os.makedirs("checkpoints", exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0
    val_losses = []
    train_losses = []
    times = []

    model_to_use = model.module if hasattr(model, "module") else model

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        start_time = time.time()

        # Train
        model.train()
        train_loss_total = 0
        n_batches = 0

        for batch_idx, batch in enumerate(train_dataloader):
            batch = move_batch_to_device(batch, DEVICE_MAP)
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
                batch = move_batch_to_device(batch, DEVICE_MAP)
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
            torch.save({'state_dict': state_dict}, f"checkpoints/{MODEL_NAME}_best.ckpt")
            print("Saved best model.")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

        # Save checkpoint per epoch
        if isinstance(model, torch.nn.DataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        torch.save({'state_dict': state_dict}, f"checkpoints/{MODEL_NAME}_epoch_{epoch}.ckpt")

        print(f"Saved checkpoint for epoch {epoch}.")

    # Save validation and training losses
    os.makedirs("results", exist_ok=True)

    with open(f"results/{MODEL_NAME}_train_losses.json", "w") as f:
        json.dump(train_losses, f)
    with open(f"results/{MODEL_NAME}_val_losses.json", "w") as f:
        json.dump(val_losses, f)
    print("Saved training and validation losses.")
    with open(f"results/{MODEL_NAME}_times.json", "w") as f:
        json.dump(times, f)
    print("Saved training times.")


if __name__ == "__main__":
    train()