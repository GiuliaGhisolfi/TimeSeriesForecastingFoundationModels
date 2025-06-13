import torch
import os

from uni2ts.loss.packed import PackedNLLLoss
from uni2ts.model.moirai import MoiraiFinetune, MoiraiModule

MODEL_PATH = "Salesforce/moirai-1.0-R-small" # "Salesforce/moirai-1.0-R-base", "Salesforce/moirai-1.0-R-large"
MODEL_NAME = "moirai_small"
DEVICE_MAP = "cuda"
EPOCHS = 10

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
        lr=model.lr,
        betas=(model.beta1, model.beta2),
        weight_decay=model.weight_decay
        )

    # Load train and validation data
    train_dataloader =
    val_dataloader =

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(EPOCHS):
        # train step
        model.train()
        for batch in train_dataloader:
            optimizer.zero_grad()
            loss = model.training_step(batch, batch_idx=0)
            loss.backward()
            optimizer.step()

        # validation step
        model.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                model.validation_step(batch, batch_idx=0)

        # Save checkpoint
        torch.save({'state_dict': model.state_dict()}, f"checkpoints/{MODEL_NAME}_epoch_{epoch}.ckpt")

if __name__ == "__main__":
    train()