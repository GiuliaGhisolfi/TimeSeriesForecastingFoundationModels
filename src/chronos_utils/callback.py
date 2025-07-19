from transformers import TrainerCallback
import json
import os

class LogCallback(TrainerCallback):
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path

        # Make directory if it does not exist
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        # Initialize the log file

        with open(self.log_file_path, "w") as f:
            f.write("[")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            with open(self.log_file_path, "a") as f:
                json.dump(logs, f)
                f.write(",\n")

    def on_train_end(self, args, state, control, **kwargs):
        with open(self.log_file_path, "a") as f:
            f.write("{}]")
