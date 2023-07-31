"""
This script is used to evaluate the different preprocessing steps on the BERT mini model.

Make sure you run the preprocessing.py script before running this script.
"""

from main import (
    ModelConfig,
    BERTClass,
    train_model,
    load_model,
    load_and_tokenize_dataset,
)
from torch import cuda
from torch.utils.data import DataLoader
import numpy as np
import torch

np.random.seed(423)
torch.backends.cudnn.benchmark = True

configs = [
    ModelConfig(
        tokenizer_model="distilbert-base-uncased",
        max_length=45,
        nn_model="prajjwal1/bert-mini",
        device="cuda" if cuda.is_available() else "cpu",
        train_batch_size=16,
        valid_batch_size=16,
        epochs=3,
        start_epoch=0,
        learning_rate=1e-05,
        dataset_type="full",
        force_reload_dataset=False,
        weight_store_template=None,
    ),
    ModelConfig(
        tokenizer_model="distilbert-base-uncased",
        max_length=45,
        nn_model="prajjwal1/bert-mini",
        device="cuda" if cuda.is_available() else "cpu",
        train_batch_size=16,
        valid_batch_size=16,
        epochs=3,
        start_epoch=0,
        learning_rate=1e-05,
        dataset_type="noemoji",
        force_reload_dataset=False,
        weight_store_template=None,
    ),
    ModelConfig(
        tokenizer_model="distilbert-base-uncased",
        max_length=45,
        nn_model="prajjwal1/bert-mini",
        device="cuda" if cuda.is_available() else "cpu",
        train_batch_size=16,
        valid_batch_size=16,
        epochs=3,
        start_epoch=0,
        learning_rate=1e-05,
        dataset_type="nostopwords",
        force_reload_dataset=False,
        weight_store_template=None,
    ),
    ModelConfig(
        tokenizer_model="distilbert-base-uncased",
        max_length=45,
        nn_model="prajjwal1/bert-mini",
        device="cuda" if cuda.is_available() else "cpu",
        train_batch_size=16,
        valid_batch_size=16,
        epochs=3,
        start_epoch=0,
        learning_rate=1e-05,
        dataset_type="nopunctuation",
        force_reload_dataset=False,
        weight_store_template=None,
    ),
    ModelConfig(
        tokenizer_model="distilbert-base-uncased",
        max_length=45,
        nn_model="prajjwal1/bert-mini",
        device="cuda" if cuda.is_available() else "cpu",
        train_batch_size=16,
        valid_batch_size=16,
        epochs=3,
        start_epoch=0,
        learning_rate=1e-05,
        dataset_type="split_hashtags",
        force_reload_dataset=False,
        weight_store_template=None,
    ),
    ModelConfig(
        tokenizer_model="distilbert-base-uncased",
        max_length=45,
        nn_model="prajjwal1/bert-mini",
        device="cuda" if cuda.is_available() else "cpu",
        train_batch_size=16,
        valid_batch_size=16,
        epochs=3,
        start_epoch=0,
        learning_rate=1e-05,
        dataset_type="spellcheck",
        force_reload_dataset=False,
        weight_store_template=None,
    ),
    ModelConfig(
        tokenizer_model="distilbert-base-uncased",
        max_length=45,
        nn_model="prajjwal1/bert-mini",
        device="cuda" if cuda.is_available() else "cpu",
        train_batch_size=16,
        valid_batch_size=16,
        epochs=3,
        start_epoch=0,
        learning_rate=1e-05,
        dataset_type="combined",
        force_reload_dataset=False,
        weight_store_template=None,
    ),
]


def main(model_config: ModelConfig):
    model = BERTClass(model=model_config.nn_model)
    model.to(model_config.device)

    dataset = load_and_tokenize_dataset(
        model_config,
        frac=1,
        train_size=0.90,
        force_reload=model_config.force_reload_dataset,
    )

    training_loader = DataLoader(
        dataset["train"],
        batch_size=model_config.train_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        dataset["validation"],
        batch_size=model_config.valid_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    print(model_config)
    train_model(
        model,
        model_config,
        training_loader,
        validation_loader,
        store_path_tmpl=model_config.weight_store_template,
    )


if __name__ == "__main__":
    for config in configs:
        main(config)
