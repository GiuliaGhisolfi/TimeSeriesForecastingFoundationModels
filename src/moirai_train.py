import json
import os

import torch
from torch.utils.data import DataLoader

from uni2ts.data.loader import PadCollate
from uni2ts.loss.packed import PackedNLLLoss
from uni2ts.model.moirai import MoiraiFinetune, MoiraiModule
from utils.moirai_utils import custom_collate_fn, get_train_and_val_datasets

MODEL_PATH = "Salesforce/moirai-1.0-R-small" # "Salesforce/moirai-1.0-R-base", "Salesforce/moirai-1.0-R-large"
MODEL_NAME = "moirai_small"
DEVICE_MAP = "cpu"

EPOCHS = 10
TEST_SIZE = 0.2
PATIENCE = 3 # Early stopping patience

def train():
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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=model._hparams['lr'],
        betas=(model._hparams['beta1'], model._hparams['beta2']),
        weight_decay=model._hparams['weight_decay']
        )

    # Load train and validation data
    train_dataset, val_dataset = get_train_and_val_datasets()

    max_length = max(len(s["target"]) for s in train_dataset)
    max_length = max(max_length, max(len(s["target"]) for s in val_dataset))
    collate_fn = PadCollate(
        seq_fields=["target"],
        target_field="target",
        pad_func_map={"target": torch.zeros},
        max_length=max_length,
    )

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    os.makedirs("checkpoints", exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0
    val_losses = []
    train_losses = []

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")

        # Train
        model.train()
        train_loss_total = 0

        for batch in train_dataloader:
            optimizer.zero_grad()
            loss = model.training_step(batch, batch_idx=0)
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item()

        train_loss_avg = train_loss_total / len(train_dataloader)
        train_losses.append(train_loss_avg)

        print(f"Epoch {epoch}: Train Loss = {train_loss_avg:.4f}")

        # Validation
        model.eval()
        val_loss_total = 0

        with torch.no_grad():
            for batch in val_dataloader:
                val_loss = model.validation_step(batch, batch_idx=0)
                val_loss_total += val_loss.item()
        
        val_loss_avg = val_loss_total / len(val_dataloader)
        val_losses.append(val_loss_avg)

        print(f"Epoch {epoch}: Val Loss = {val_loss_avg:.4f}")

        # Early stopping
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            patience_counter = 0
            # Save best model
            torch.save({'state_dict': model.state_dict()}, f"checkpoints/{MODEL_NAME}_best.ckpt")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

        # Save checkpoint per epoch
        torch.save({'state_dict': model.state_dict()}, f"checkpoints/{MODEL_NAME}_epoch_{epoch}.ckpt")

    # Save validation and training losses
    os.makedirs("results", exist_ok=True)

    with open("results/{MODEL_NAME}_train_losses.json", "w") as f:
        json.dump(train_losses, f)
    with open("results/{MODEL_NAME}_val_losses.json", "w") as f:
        json.dump(val_losses, f)


if __name__ == "__main__":
    train()