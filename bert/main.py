from dataset import load_dataset, tokenize_dataset, load_and_tokenize_dataset
import transformers
import torch
from typing import NamedTuple
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
import copy
from torch import cuda
import pickle
from pprint import pprint
import pandas as pd

np.random.seed(1)


class ModelConfig(NamedTuple):
    tokenizer_model: str
    max_length: int
    nn_model: str
    device: str
    train_batch_size: int
    valid_batch_size: int
    epochs: int
    learning_rate: float
    dataset_type: str
    force_reload_dataset: bool

    def __repr__(self) -> str:
        out = "Best Transformer Config:\n"
        for k, v in zip(self._fields, self):
            out += f"- {k:<16} = {v}\n"
        return out


class BERTClass(torch.nn.Module):
    def __init__(self, model: str):
        match model:
            case "prajjwal1/bert-small":
                width = 512
            case "prajjwal1/bert-mini":
                width = 256
            case "prajjwal1/bert-tiny":
                width = 128

        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained(model, return_dict=False)
        self.l2 = torch.nn.Dropout(0.5)
        self.l3 = torch.nn.Linear(width, 1)
        self.l4 = torch.nn.Sigmoid()

    def forward(self, ids, mask, token_type_ids):
        _, output = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output = self.l2(output)
        output = self.l3(output)
        output = self.l4(output)
        return output


def train_model(model, model_config: ModelConfig, train_data, val_data):
    """
    Train the model
    """

    loss_fn = torch.nn.BCELoss()  # binary cross entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=model_config.learning_rate)
    device = model_config.device

    # Hold the best model
    best_acc = -np.inf  # init to negative infinity
    best_weights = None

    for epoch in range(model_config.epochs):
        model.train()
        with tqdm(
            train_data,
            unit="batch",
            mininterval=0,
            miniters=200,
        ) as bar:
            bar.set_description(f"Epoch {epoch}")

            correct_cnt = 0
            cnt = 0
            for i, batch in enumerate(bar):
                # take a batch

                # forward pass
                y_batch = batch["label"].to(device, dtype=torch.float32).reshape(-1, 1)
                y_pred = model(
                    batch["input_ids"].to(device, dtype=torch.long),
                    batch["attention_mask"].to(device, dtype=torch.long),
                    batch["token_type_ids"].to(device, dtype=torch.long),
                )
                loss = loss_fn(y_pred, y_batch)
                cnt += len(y_batch)
                correct_cnt += int((y_pred.round() == y_batch).int().sum())

                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # print progress

                if i % 200 == 0:
                    acc = (y_pred.round() == y_batch).float().mean()
                    bar.set_postfix(
                        loss=f"{float(loss):.3f}",
                        acc=f"{float(acc):.3f}",
                        total_acc=f"{correct_cnt/cnt:.3f}",
                    )

            acc = correct_cnt / cnt
            print(f"Epoch {epoch} training accuracy: {acc:.3f}")

        # evaluate accuracy at end of each epoch
        model.eval()
        with torch.no_grad():
            correct_cnt = 0
            cnt = 0
            for val_batch in tqdm(val_data, desc="Validation"):
                y_val = (
                    val_batch["label"].to(device, dtype=torch.float32).reshape(-1, 1)
                )
                y_pred = model(
                    val_batch["input_ids"].to(device, dtype=torch.long),
                    val_batch["attention_mask"].to(device, dtype=torch.long),
                    val_batch["token_type_ids"].to(device, dtype=torch.long),
                )
                cnt += len(y_val)
                correct_cnt += int((y_pred.round() == y_val).int().sum())

        acc = correct_cnt / cnt
        print(f"Epoch {epoch} validation accuracy: {acc:.3f}")
        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(model.state_dict())

    # restore model and return best accuracy
    model.load_state_dict(best_weights)
    return best_acc


def eval_model(model, model_config: ModelConfig, train_data, val_data):
    """
    Evaluate model on train and validation data
    """

    device = model_config.device
    model.eval()
    with torch.no_grad():
        correct_cnt = 0
        cnt = 0
        for batch in tqdm(val_data):
            y = batch["label"].to(device, dtype=torch.float32).reshape(-1, 1)
            y_pred = model(
                batch["input_ids"].to(device, dtype=torch.long),
                batch["attention_mask"].to(device, dtype=torch.long),
                batch["token_type_ids"].to(device, dtype=torch.long),
            )
            cnt += len(y)
            correct_cnt += int((y_pred.round() == y).int().sum())

        acc = correct_cnt / cnt
        print(f"validation accuracy: {acc:.3f}")

        correct_cnt = 0
        cnt = 0
        for batch in tqdm(train_data):
            y = batch["label"].to(device, dtype=torch.float32).reshape(-1, 1)
            y_pred = model(
                batch["input_ids"].to(device, dtype=torch.long),
                batch["attention_mask"].to(device, dtype=torch.long),
                batch["token_type_ids"].to(device, dtype=torch.long),
            )
            cnt += len(y)
            correct_cnt += int((y_pred.round() == y).int().sum())

        acc = correct_cnt / cnt
        print(f"training accuracy: {acc:.3f}")


def generate_predictions(model, model_config: ModelConfig, test_data, output_file):
    """
    Generate predictions for test data that can be submitted to Kaggle
    """
    model.eval()
    device = model_config.device
    with torch.no_grad():
        ids = test_data["id"].reshape(-1, 1)
        y_pred = model(
            test_data["input_ids"].to(device, dtype=torch.long),
            test_data["attention_mask"].to(device, dtype=torch.long),
            test_data["token_type_ids"].to(device, dtype=torch.long),
        ).round()

        # move tensors to cpu
        ids = ids.cpu()
        y_pred = y_pred.cpu()

        df = pd.DataFrame({"Id": ids.squeeze(), "Prediction": y_pred.squeeze()})
        df["Prediction"] = df["Prediction"].apply(lambda x: 1 if x == 1 else -1)
        df.to_csv(output_file, index=False, sep=",", header=True)

        print(f"generated predictions ({output_file=})")


def save_model(model, path):
    """
    Save the model to the given path
    """
    torch.save(model.state_dict(), path)


def load_model(model, path):
    """
    Load the model from the given path
    """
    print(f"Loading model from {path}")
    model.load_state_dict(torch.load(path))


def main():
    model_config = ModelConfig(
        tokenizer_model="distilbert-base-uncased",
        max_length=45,
        nn_model="prajjwal1/bert-mini",
        device="cuda" if cuda.is_available() else "cpu",
        train_batch_size=64,
        valid_batch_size=64,
        epochs=6,
        learning_rate=1e-04,
        dataset_type="combined",
        force_reload_dataset=False,
    )

    print(model_config)

    model = BERTClass(model=model_config.nn_model)
    model.to(model_config.device)

    dataset = load_and_tokenize_dataset(
        model_config,
        frac=1,
        train_size=0.8,
        force_reload=model_config.force_reload_dataset,
    )

    training_loader = DataLoader(
        dataset["train"],
        batch_size=model_config.train_batch_size,
        shuffle=True,
        num_workers=0,
    )
    validation_loader = DataLoader(
        dataset["validation"],
        batch_size=model_config.valid_batch_size,
        shuffle=False,
        num_workers=0,
    )

    load_model(model, "bert_mini_4_epoch_combined_8.pkl")
    train_model(model, model_config, training_loader, validation_loader)
    save_model(model, "bert_mini_10_epoch_combined_8.pkl")

    # eval_model(model, model_config, training_loader, validation_loader)

    # train_model(model, model_config, training_loader, validation_loader)
    # save_model(model, "bert_mini_4_epoch_combined_8.pkl")

    # eval_model(model, model_config, training_loader, validation_loader)

    # generate_predictions(
    #     model, model_config, dataset["test"], "bert_mini_3_epoch_full_results.csv"
    # )


if __name__ == "__main__":
    main()
