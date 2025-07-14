import argparse
import os
import numpy as np
import pandas as pd
import torch
from chronos.chronos_bolt import ChronosBoltPipeline
from chronos.chronos import ChronosModel, ChronosConfig
from tqdm import tqdm
from chronos_utils.chronos_dataset import ChronosDataset, has_enough_observations
import logging
from transformers import (
    Trainer,
    TrainingArguments,
    PretrainedConfig
)
from gluonts.dataset.common import FileDataset
from gluonts.itertools import Filter
from functools import partial
from pathlib import Path
import yaml


DATA_PATH = "/raid/decaro/TimeSeriesForecastingFoundationModels/data/" # "data/"
OUTPUT_DIR = "/raid/decaro/TimeSeriesForecastingFoundationModels/chronos_output"

MODEL_NAME = "chronos-bolt-tiny" # "chronos-bolt-mini", "chronos-bolt-small", "chronos-bolt-base"
MODEL_MAP = {
    "chronos-bolt-tiny": "autogluon/chronos-bolt-tiny",
    "chronos-bolt-mini": "autogluon/chronos-bolt-mini",
    "chronos-bolt-small": "autogluon/chronos-bolt-small",
    "chronos-bolt-base": "autogluon/chronos-bolt-base",
}

EPOCHS = 10
RANDOM_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
    model_name=MODEL_NAME,
    epochs=EPOCHS,
    batch_size=2,
    learning_rate=1e-5,

    context_length: int = 512,
    prediction_length: int = 64,
    model_type: str = "seq2seq",
    tokenizer_class: str = "MeanScaleUniformBins",
    tokenizer_kwargs: dict = {'low_limit': -15.0, 'high_limit': 15.0},
    n_tokens: int = 4096,
    n_special_tokens: int = 2,
    pad_token_id: int = 0,
    eos_token_id: int = 1,
    use_eos_token: bool = True,
    num_samples: int = 20,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 1.0,    
    ):
    # Initialize ChronosModel
    model_path = MODEL_MAP[model_name]
    pipeline = ChronosBoltPipeline.from_pretrained(model_path)
    
    with open("src/chronos_configs/chronos-bolt-tiny.yaml", "r") as f:
        config_dict = yaml.safe_load(f)
    config = PretrainedConfig(**config_dict)

    model = ChronosModel(
        config=config,
        model=pipeline,#model_path,
    )
    print("Model loaded.")

    chronos_config = ChronosConfig(
        tokenizer_class=tokenizer_class,
        tokenizer_kwargs=tokenizer_kwargs,
        n_tokens=n_tokens,
        n_special_tokens=n_special_tokens,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        use_eos_token=use_eos_token,
        model_type=model_type,
        context_length=context_length,
        prediction_length=prediction_length,
        num_samples=num_samples,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    # Add extra items to model config so that it's saved in the ckpt
    model.config.chronos_config = chronos_config.__dict__

    # Data
    training_data_paths = [
        DATA_PATH+"dummy_dataset_gluonts.jsonl"
        #DATA_PATH+"dataset_gluonts.jsonl"
        ] # FIXME
    train_datasets = [
        Filter(
            partial(
                has_enough_observations,
                min_length= 64 + prediction_length,
                max_missing_prop=0.9,
            ),
            FileDataset(path=Path(data_path), freq="h"),
        )
        for data_path in training_data_paths
    ]

    shuffled_train_dataset = ChronosDataset(
        datasets=train_datasets,
        probabilities=[1 / len(train_datasets)] * len(train_datasets),
        tokenizer=chronos_config.create_tokenizer(),
        mode="training",
    ).shuffle(shuffle_buffer_length=100)

    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("PyTorch sees CUDA:", torch.cuda.is_available())

    # Define training args
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        lr_scheduler_type="linear",
        warmup_ratio=0.0,
        optim="adamw_torch_fused",
        logging_dir=OUTPUT_DIR+"/logs",
        logging_strategy="steps",
        logging_steps=50,
        save_strategy="steps",
        save_steps=5000,
        report_to=["tensorboard"],
        max_steps=10000,
        gradient_accumulation_steps=2,
        dataloader_num_workers=1,
        torch_compile=True,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )

    # Fit the model with train and validation data
    print(">> START TRAINING ...")

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=shuffled_train_dataset,
    )

    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=1)#EPOCHS) #FIXME
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)

    args = parser.parse_args()

    train(
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
