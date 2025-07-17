import os
import time

import orjson
from lightning.pytorch.callbacks import Callback


class EpochStatsLogger(Callback):
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        self.train_losses = {
            "PackedNLLLoss": [],
        }
        self.val_losses = {
            "PackedNLLLoss": [],
            "PackedNMAELoss": [],
            "PackedNMSELoss": [],
            "PackedMAPELoss": [],
        }
        self.epoch_times = []
        os.makedirs("results_moirai", exist_ok=True)

    def on_train_epoch_start(self, trainer, pl_module):
        self._start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        elapsed = time.time() - self._start_time
        self.epoch_times.append(elapsed)

        # train loss
        train_loss = trainer.callback_metrics.get("train/PackedNLLLoss")
        if train_loss is not None:
            self.train_losses["PackedNLLLoss"].append(train_loss.item())

        # val loss
        val_loss_NLL = trainer.callback_metrics.get("val/PackedNLLLoss")
        val_loss_NMAE = trainer.callback_metrics.get("val/PackedNMAELoss")
        val_loss_NMSE = trainer.callback_metrics.get("val/PackedNMSELoss")
        val_loss_MAPE = trainer.callback_metrics.get("val/PackedMAPELoss")

        if val_loss_NLL is not None:
            self.val_losses["PackedNLLLoss"].append(val_loss_NLL.item())
        if val_loss_NMAE is not None:
            self.val_losses["PackedNMAELoss"].append(val_loss_NMAE.item())
        if val_loss_NMSE is not None:
            self.val_losses["PackedNMSELoss"].append(val_loss_NMSE.item())
        if val_loss_MAPE is not None:
            self.val_losses["PackedMAPELoss"].append(val_loss_MAPE.item())

        msg = f"Epoch {trainer.current_epoch + 1} completed in {elapsed:.2f}s"
        if train_loss is not None:
            msg += f" — Train Loss: {train_loss.item():.4f}"
        if val_loss_NLL is not None:
            msg += f" — Val NLL Loss: {val_loss_NLL.item():.4f}"
        if val_loss_NMAE is not None:
            msg += f" — Val NMAE Loss: {val_loss_NMAE.item():.4f}"
        if val_loss_NMSE is not None:
            msg += f" — Val NMSE Loss: {val_loss_NMSE.item():.4f}"
        if val_loss_MAPE is not None:
            msg += f" — Val MAPE Loss: {val_loss_MAPE.item():.4f}"
        print(msg)

        # Save losses and times to disk after each epoch
        with open(f"results_moirai/{self.model_name}_train_losses.json", "wb") as f:
            f.write(orjson.dumps(self.train_losses))
        with open(f"results_moirai/{self.model_name}_val_losses.json", "wb") as f:
            f.write(orjson.dumps(self.val_losses))
        with open(f"results_moirai/{self.model_name}_times.json", "wb") as f:
            f.write(orjson.dumps(self.epoch_times))
