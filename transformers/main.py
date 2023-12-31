"""
This script can run the following transformers models:

- BERT mini
- RoBERTa base
- Emoji RoBERTa
- Twitter RoBERTa
- Twitter RoBERTa EN

You can specify which model to run by changing the `model_config` variable in the main function.
"""
from dataset import (
    load_dataset,
    tokenize_dataset,
    load_and_tokenize_dataset,
    get_obervation_dataset,
)
import transformers
from transformers import RobertaModel, AutoModelForSequenceClassification
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

np.random.seed(423)
torch.backends.cudnn.benchmark = True


class ModelConfig(NamedTuple):
    tokenizer_model: str
    max_length: int
    nn_model: str
    device: str
    train_batch_size: int
    valid_batch_size: int
    epochs: int
    start_epoch: int
    learning_rate: float
    dataset_type: str
    force_reload_dataset: bool
    weight_store_template: str

    def __repr__(self) -> str:
        out = "Best Transformer Config:\n"
        for k, v in zip(self._fields, self):
            out += f"- {k:<20} = {v}\n"
        return out


BERT_MINI = ModelConfig(
    tokenizer_model="prajjwal1/bert-mini",
    max_length=45,
    nn_model="prajjwal1/bert-mini",
    device="cuda" if cuda.is_available() else "cpu",
    train_batch_size=32,
    valid_batch_size=32,
    epochs=3,
    start_epoch=0,
    learning_rate=1e-05,
    dataset_type="combined",
    force_reload_dataset=True,
    weight_store_template="mini_bert_{}_epoch_combined_final.pkl",
)

ROBERTA_BASE = ModelConfig(
    tokenizer_model="roberta-base",
    max_length=45,
    nn_model="roberta-base",
    device="cuda" if cuda.is_available() else "cpu",
    train_batch_size=32,
    valid_batch_size=32,
    epochs=3,
    start_epoch=0,
    learning_rate=1e-05,
    dataset_type="combined",
    force_reload_dataset=True,
    weight_store_template="base_roberta_{}_epoch_combined.pkl",
)

EMOJI_ROBERTA = ModelConfig(
    tokenizer_model="cardiffnlp/roberta-base-emoji",
    max_length=45,
    nn_model="cardiffnlp/roberta-base-emoji",
    device="cuda" if cuda.is_available() else "cpu",
    train_batch_size=32,
    valid_batch_size=32,
    epochs=3,
    start_epoch=0,
    learning_rate=1e-05,
    dataset_type="combined",
    force_reload_dataset=True,
    weight_store_template="emoji_roberta_{}_epoch_combined.pkl",
)

TWITTER_ROBERTA = ModelConfig(
    tokenizer_model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    max_length=45,
    nn_model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    device="cuda" if cuda.is_available() else "cpu",
    train_batch_size=32,
    valid_batch_size=32,
    epochs=3,
    start_epoch=0,
    learning_rate=1e-05,
    dataset_type="combined",
    force_reload_dataset=True,
    weight_store_template="twitter_roberta_{}_epoch_combined.pkl",
)

TWITTER_ROBERTA_EN = ModelConfig(
    tokenizer_model="cardiffnlp/roberta-base-tweet-sentiment-en",
    max_length=45,
    nn_model="cardiffnlp/roberta-base-tweet-sentiment-en",
    device="cuda" if cuda.is_available() else "cpu",
    train_batch_size=32,
    valid_batch_size=32,
    epochs=3,
    start_epoch=0,
    learning_rate=1e-05,
    dataset_type="combined",
    force_reload_dataset=True,
    weight_store_template="twitter_en_roberta_{}_epoch_combined.pkl",
)


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
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(width, 1)
        self.l4 = torch.nn.Sigmoid()

    def forward(self, ids, mask, token_type_ids):
        _, output = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output = self.l2(output)
        output = self.l3(output)
        output = self.l4(output)
        return output


class RoBERTaClass(torch.nn.Module):
    def __init__(self):
        super(RoBERTaClass, self).__init__()
        self.l1 = RobertaModel.from_pretrained("roberta-base", return_dict=False)

        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 1)
        self.l4 = torch.nn.Sigmoid()

    def forward(self, ids, mask, token_type_ids):
        _, output = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output = self.l2(output)
        output = self.l3(output)
        output = self.l4(output)
        return output


class RoBERTaTwitter(torch.nn.Module):
    def __init__(self):
        print("using RoBERTaTwitter")
        super(RoBERTaTwitter, self).__init__()
        self.l1 = AutoModelForSequenceClassification.from_pretrained(
            "cardiffnlp/twitter-roberta-base-sentiment-latest",
            # output_hidden_states=True,
            return_dict=True,
        )

        # self.l1 = torch.nn.Sequential(*list(self.l1.children())[:-2])

        # self.l2 = torch.nn.Linear(768, 768)
        # self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(3, 1)
        self.l4 = torch.nn.Sigmoid()

    def forward(self, ids, mask, token_type_ids):
        output = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        # print(dir(output))
        # print(output.hidden_states[-1][:, -1, :].shape)
        # output = torch.mean(output[0], 1)

        # output = self.l2(output.hidden_states[-1][:, -1, :])
        output = self.l3(output.logits)
        output = self.l4(output)
        return output


class RoBERTaTwitterEN(torch.nn.Module):
    def __init__(self):
        print("using RoBERTaTwitterEN")
        super(RoBERTaTwitterEN, self).__init__()
        self.l1 = AutoModelForSequenceClassification.from_pretrained(
            "cardiffnlp/roberta-base-tweet-sentiment-en",
            output_hidden_states=True,
            return_dict=True,
        )

        # self.l2 = torch.nn.Linear(3, 3)
        # self.l3 = torch.nn.RReLU()
        self.l4 = torch.nn.Linear(3, 1)
        self.l5 = torch.nn.Sigmoid()

    def forward(self, ids, mask, token_type_ids):
        output = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)

        # output = self.l2(output[0])
        # output = self.l3(output)
        output = self.l4(output[0])
        output = self.l5(output)
        return output


class RoBERTaEmoji(torch.nn.Module):
    def __init__(self):
        print("using RoBERTaEmoji")
        super(RoBERTaEmoji, self).__init__()
        self.l1 = AutoModelForSequenceClassification.from_pretrained(
            "cardiffnlp/roberta-base-emoji",
            # output_hidden_states=True,
            return_dict=True,
        )

        # self.l1 = torch.nn.Sequential(*list(self.l1.children())[:-2])

        self.l2 = torch.nn.Linear(20, 20)
        self.l3 = torch.nn.ReLU()
        self.l4 = torch.nn.Linear(20, 1)
        self.l5 = torch.nn.Sigmoid()

    def forward(self, ids, mask, token_type_ids):
        output = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)

        output = self.l2(output.logits)
        output = self.l3(output)
        output = self.l4(output)
        output = self.l5(output)
        return output


class RoBERTaIrony(torch.nn.Module):
    def __init__(self):
        print("using RoBERTaIrony")
        super(RoBERTaIrony, self).__init__()
        self.l1 = AutoModelForSequenceClassification.from_pretrained(
            "cardiffnlp/twitter-roberta-base-sentiment-latest",
            output_hidden_states=True,
            return_dict=True,
        )

        # more complex
        self.l2 = torch.nn.Linear(770, 770)
        self.l3 = torch.nn.RReLU()
        self.l4 = torch.nn.Dropout(0.3)
        self.l5 = torch.nn.Linear(770, 1)
        self.l6 = torch.nn.Sigmoid()

        # self.l2 = torch.nn.Linear(5, 5)
        # self.l3 = torch.nn.RReLU()
        # self.l4 = torch.nn.Linear(5, 1)
        # self.l5 = torch.nn.Sigmoid()

    def forward(self, ids, mask, token_type_ids, irony):
        output = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)

        # more complex
        # hidden_states = output.hidden_states[-1][:, -1, :]
        # output = torch.cat((output.logits, irony), 1)

        # print(output.logits.shape)

        output = output.hidden_states[-1][:, -1, :]

        # print(output.shape)

        irony = torch.nn.functional.pad(
            irony, (0, 0, 0, output.shape[0] - irony.shape[0]), "constant", 0
        )

        output = torch.cat((output, irony), 1)

        output = self.l2(output)
        output = self.l3(output)
        output = self.l4(output)
        output = self.l5(output)
        output = self.l6(output)
        return output

        # output = torch.cat((output.logits, irony), 1)
        # output = self.l3(output)
        # output = self.l4(output)
        # return output


class RoBERTaXLM(torch.nn.Module):
    def __init__(self):
        super(RoBERTaXLM, self).__init__()
        self.l1 = RobertaModel.from_pretrained(
            "cardiffnlp/twitter-xlm-roberta-base-sentiment", return_dict=False
        )

        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 1)
        self.l4 = torch.nn.Sigmoid()

    def forward(self, ids, mask, token_type_ids):
        _, output = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output = self.l2(output)
        output = self.l3(output)
        output = self.l4(output)
        return output


def train_model(
    model,
    model_config: ModelConfig,
    train_data,
    val_data,
    *,
    store_path_tmpl=None,
):
    """
    Train the model
    """

    print(
        f"Training model for {model_config.epochs} epochs starting at {model_config.start_epoch}"
    )

    loss_fn = torch.nn.BCELoss()  # binary cross entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=model_config.learning_rate)
    device = model_config.device

    # Hold the best model
    best_acc = -np.inf  # init to negative infinity
    best_weights = None

    for epoch in range(model_config.start_epoch, model_config.epochs):
        model.train()
        with tqdm(
            train_data,
            unit="batch",
            # mininterval=5,
            # maxinterval=10,
            miniters=50,
        ) as bar:
            bar.set_description(f"Epoch {epoch}")

            correct_cnt = 0
            cnt = 0
            for i, batch in enumerate(bar):
                # forward pass
                y_batch = batch["label"].to(device, dtype=torch.float32).reshape(-1, 1)
                y_pred = model(
                    batch["input_ids"].to(device, dtype=torch.long),
                    batch["attention_mask"].to(device, dtype=torch.long),
                    batch["token_type_ids"].to(device, dtype=torch.long),
                    # batch["irony"].to(device, dtype=torch.float32),
                )
                loss = loss_fn(y_pred, y_batch)

                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # print progress

                cnt += len(y_batch)
                correct_cnt += int((y_pred.round() == y_batch).int().sum())

                if i % 50 == 0:
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
                    # batch["irony"].to(device, dtype=torch.float32),
                )
                cnt += len(y_val)
                correct_cnt += int((y_pred.round() == y_val).int().sum())

        acc = correct_cnt / cnt
        print(f"Epoch {epoch} validation accuracy: {acc:.3f}\n")

        if store_path_tmpl is not None:
            save_model(model, store_path_tmpl.format(epoch))

        # if acc > best_acc:
        #     best_acc = acc
        #     best_weights = copy.deepcopy(model.state_dict())

    # restore model and return best accuracy
    # model.load_state_dict(best_weights)
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


def observe_model(model, model_config: ModelConfig):
    dataset = get_obervation_dataset(model_config, frac=1, train_size=0.85)

    validation_loader = DataLoader(
        dataset["validation"],
        batch_size=model_config.valid_batch_size,
        shuffle=False,
        num_workers=0,
    )

    correct_output_file = "correct.csv"
    incorrect_output_file = "incorrect.csv"

    correct_file = open(correct_output_file, "w")
    incorrect_file = open(incorrect_output_file, "w")
    model.eval()
    device = model_config.device
    with torch.no_grad():
        correct_cnt = 0
        cnt = 0
        for val_batch in tqdm(validation_loader, desc="Validation"):
            y_val = val_batch["label"].to(device, dtype=torch.float32).reshape(-1, 1)
            y_pred = model(
                val_batch["input_ids"].to(device, dtype=torch.long),
                val_batch["attention_mask"].to(device, dtype=torch.long),
                val_batch["token_type_ids"].to(device, dtype=torch.long),
            )
            cnt += len(y_val)
            correct = (y_pred.round() == y_val).int().cpu()
            correct_cnt += int((y_pred.round() == y_val).int().sum())

            for i, c in enumerate(correct):
                if c == 1:
                    print(
                        f"{int(val_batch['label'][i])}: {val_batch['tweet'][i]}",
                        file=correct_file,
                    )
                elif c == 0:
                    print(
                        f"{int(val_batch['label'][i])}: {val_batch['tweet'][i]}",
                        file=incorrect_file,
                    )
                else:
                    raise ValueError("incorrect value for correct")

        print(f"correct: {correct_cnt/cnt:.3f}")
        correct_file.close()
        incorrect_file.close()


def generate_predictions(model, model_config: ModelConfig, test_data, output_file):
    """
    Generate predictions for test data that can be submitted to Kaggle
    """
    model.eval()
    device = model_config.device

    output = pd.DataFrame(columns=["Id", "Prediction"])

    with torch.no_grad():
        for batch in tqdm(test_data, desc="Generating predictions"):
            ids = batch["id"].reshape(-1, 1)
            y_pred = model(
                batch["input_ids"].to(device, dtype=torch.long),
                batch["attention_mask"].to(device, dtype=torch.long),
                batch["token_type_ids"].to(device, dtype=torch.long),
            ).round()

            # move tensors to cpu
            ids = ids.cpu()
            y_pred = y_pred.cpu()

            df = pd.DataFrame({"Id": ids.squeeze(), "Prediction": y_pred.squeeze()})
            df["Prediction"] = df["Prediction"].apply(lambda x: 1 if x == 1 else -1)

            output = pd.concat([output, df])

    output.to_csv(output_file, index=False, sep=",", header=True)
    print(f"generated predictions ({output_file=})")


def generate_val_predictions(model, model_config: ModelConfig, val_data, output_file):
    """
    Generate predictions for test data that can be submitted to Kaggle
    """
    model.eval()
    device = model_config.device

    output = pd.DataFrame(columns=["Prediction", "Label", "Tweet"])

    with torch.no_grad():
        for batch in tqdm(val_data, desc="Generating validation predictions"):
            labels = batch["label"].reshape(-1, 1)
            tweets = batch["tweet"]
            y_pred = model(
                batch["input_ids"].to(device, dtype=torch.long),
                batch["attention_mask"].to(device, dtype=torch.long),
                batch["token_type_ids"].to(device, dtype=torch.long),
            ).round()

            # move tensors to cpu
            labels = labels.cpu()
            y_pred = y_pred.cpu()

            df = pd.DataFrame(
                {
                    "Prediction": y_pred.squeeze(),
                    "Label": labels.squeeze(),
                    "Tweet": tweets,
                }
            )
            df["Prediction"] = df["Prediction"].apply(lambda x: 1 if x == 1 else -1)
            df["Label"] = df["Label"].apply(lambda x: 1 if x == 1 else -1)

            output = pd.concat([output, df])

    output.to_csv(output_file, index=False, sep="\t", header=True)
    print(f"generated validation predictions ({output_file=})")


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
    # choose the model you want to use here
    # model_config = BERT_MINI
    # model_config = ROBERTA_BASE
    # model_config = EMOJI_ROBERTA
    model_config = TWITTER_ROBERTA
    # model_config = TWITTER_ROBERTA_EN

    print(model_config)
    match model_config.nn_model:
        case "cardiffnlp/roberta-base-tweet-sentiment-en":
            model = RoBERTaTwitterEN()
        case "cardiffnlp/roberta-base-emoji":
            model = RoBERTaEmoji()
        case "irony":
            model = RoBERTaIrony()
        case "cardiffnlp/twitter-roberta-base-sentiment-latest":
            model = RoBERTaTwitter()
        case "cardiffnlp/twitter-xlm-roberta-base-sentiment":
            model = RoBERTaXLM()
        case "roberta-base":
            model = RoBERTaClass()
        case _:  # bert
            model = BERTClass(model=model_config.nn_model)
    model.to(model_config.device)

    # load the dataset
    dataset = load_and_tokenize_dataset(
        model_config,
        frac=1,
        train_size=0.90,
        force_reload=model_config.force_reload_dataset,
        include_tweet=True,
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
    test_loader = DataLoader(
        dataset["test"],
        batch_size=model_config.valid_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # train the model
    # load_model(model, "twitter_roberta_2_epoch_combined.pkl")
    train_model(
        model,
        model_config,
        training_loader,
        validation_loader,
        store_path_tmpl=model_config.weight_store_template,
    )

    generate_predictions(
        model,
        model_config,
        test_loader,
        "submission.csv",
    )


if __name__ == "__main__":
    main()
