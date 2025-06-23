import os
import time

import orjson
from lightning.pytorch.callbacks import Callback


class EpochStatsLogger(Callback):
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        self.train_losses = []
        self.val_losses = []
        self.epoch_times = []
        os.makedirs("results", exist_ok=True)

    def on_train_epoch_start(self, trainer, pl_module):
        self._start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        elapsed = time.time() - self._start_time
        self.epoch_times.append(elapsed)

        # train loss
        train_loss = trainer.callback_metrics.get("train/PackedNLLLoss")
        if train_loss is not None:
            self.train_losses.append(train_loss.item())

        # val loss
        val_loss = trainer.callback_metrics.get("val/PackedNLLLoss")
        if val_loss is not None:
            self.val_losses.append(val_loss.item())

        # Save losses and times to disk after each epoch
        with open(f"results/{self.model_name}_train_losses.json", "wb") as f:
            f.write(orjson.dumps(self.train_losses))
        with open(f"results/{self.model_name}_val_losses.json", "wb") as f:
            f.write(orjson.dumps(self.val_losses))
        with open(f"results/{self.model_name}_times.json", "wb") as f:
            f.write(orjson.dumps(self.epoch_times))
