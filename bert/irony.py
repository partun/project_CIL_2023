from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import urllib.request
from torch import cuda
from typing import NamedTuple
import pandas as pd
import torch
from datasets import Features, ClassLabel, Value, Dataset, DatasetDict
from transformers import AutoTokenizer, RobertaTokenizer
from torch.utils.data import DataLoader
import os
from tqdm import tqdm


class ModelConfig(NamedTuple):
    tokenizer_model: str
    max_length: int
    nn_model: str
    device: str
    dataset_type: str
    force_reload_dataset: bool

    def __repr__(self) -> str:
        out = "Best Transformer Config:\n"
        for k, v in zip(self._fields, self):
            out += f"- {k:<16} = {v}\n"
        return out


def get_preprocess(tokenizer, model_config, *, include_tweet=False):
    def preprocess(examples):
        input = tokenizer(
            examples["tweet"],
            truncation=True,
            max_length=model_config.max_length,
            padding="max_length",
            pad_to_max_length=True,
            return_token_type_ids=True,
        )

        input_ids = torch.tensor(input["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(input["attention_mask"], dtype=torch.long)
        token_type_ids = torch.tensor(input["token_type_ids"], dtype=torch.long)

        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "tweet": examples["tweet"],
        }
        if "id" in examples:
            output["id"] = examples["id"]
        return output

    return preprocess


def load_dataset(model_config: ModelConfig, frac=1):
    neg_path = "../twitter-datasets/train_neg_full_combined.csv"
    pos_path = "../twitter-datasets/train_pos_full_combined.csv"
    test_path = "../twitter-datasets/test_data_combined.csv"

    with open(neg_path, "r") as neg_file:
        tweets_neg = pd.read_csv(
            neg_file,
            sep="\t",
            lineterminator="\n",
            encoding="utf8",
            names=["tweet"],
            quoting=3,
        )
        neg_file.seek(0)
        assert len(tweets_neg) == len(neg_file.readlines())

    with open(pos_path, "r") as pos_file:
        tweets_pos = pd.read_csv(
            pos_file,
            sep="\t",
            lineterminator="\n",
            encoding="utf8",
            names=["tweet"],
            quoting=3,
        )
        pos_file.seek(0)
        assert len(tweets_pos) == len(pos_file.readlines())

    with open(test_path, "r") as test_file:
        test_dataset = pd.read_csv(
            test_file,
            sep="\t",
            lineterminator="\n",
            encoding="utf8",
            names=["id", "tweet"],
            header=None,
            quoting=3,
        )
        test_file.seek(0)
        assert len(test_dataset) == len(test_file.readlines())

    if frac < 1:
        tweets_pos = tweets_pos.sample(frac=frac).reset_index(drop=True)
        tweets_neg = tweets_neg.sample(frac=frac).reset_index(drop=True)
        test_dataset = test_dataset.sample(frac=frac).reset_index(drop=True)

    print("Dataset Head")
    print(tweets_pos.head())

    dataset = DatasetDict(
        {
            "pos": Dataset.from_pandas(
                tweets_pos,
                features=Features({"tweet": Value("string")}),
            ),
            "neg": Dataset.from_pandas(
                tweets_neg,
                features=Features({"tweet": Value("string")}),
            ),
            "test": Dataset.from_pandas(
                test_dataset,
                features=Features({"tweet": Value("string"), "id": Value("int64")}),
            ),
        }
    )

    dataset = dataset.with_format("torch")

    tokenizer = AutoTokenizer.from_pretrained(model_config.tokenizer_model)
    encoded_dataset = dataset.map(
        get_preprocess(tokenizer, model_config),
        batched=True,
    )

    print("tokenized dataset successfully!")
    print(encoded_dataset)
    return encoded_dataset


# output = model(**encoded_input)
# scores = output[0][0].detach().numpy()
# scores = softmax(scores)


# ranking = np.argsort(scores)
# ranking = ranking[::-1]
# for i in range(scores.shape[0]):
#     l = labels[ranking[i]]
#     s = scores[ranking[i]]
#     print(f"{i+1}) {l} {np.round(float(s), 4)}")


def predict(model, dataloader, model_config, output_path, include_ids=False):
    output_file = open(output_path, "w")
    device = model_config.device

    with torch.no_grad():
        for batch in tqdm(dataloader):
            y_pred = model(
                batch["input_ids"].to(device, dtype=torch.long),
                batch["attention_mask"].to(device, dtype=torch.long),
                batch["token_type_ids"].to(device, dtype=torch.long),
            )

            scores = torch.softmax(y_pred[0][:], axis=1).cpu().detach().numpy()

            if include_ids:
                for i, (tid, tweet) in enumerate(zip(batch["id"], batch["tweet"])):
                    print(
                        f"{tid:.0f}\t{tweet}\t{scores[i][0]:.4f}\t{scores[i][1]:.4f}",
                        file=output_file,
                    )
            else:
                for i, tweet in enumerate(batch["tweet"]):
                    print(
                        f"{tweet}\t{scores[i][0]:.4f}\t{scores[i][1]:.4f}",
                        file=output_file,
                    )

        output_file.close()


def main():
    model_config = ModelConfig(
        tokenizer_model="cardiffnlp/twitter-roberta-base-irony",
        max_length=45,
        nn_model="cardiffnlp/twitter-roberta-base-irony",
        device="cuda" if cuda.is_available() else "cpu",
        dataset_type="combined",
        force_reload_dataset=False,
    )

    dataset = load_dataset(
        model_config,
        frac=1,
    )

    pos_loader = DataLoader(
        dataset["pos"],
        batch_size=128,
        shuffle=False,
        num_workers=0,
    )
    neg_loader = DataLoader(
        dataset["neg"],
        batch_size=128,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        dataset["test"],
        batch_size=128,
        shuffle=False,
        num_workers=0,
    )

    model = AutoModelForSequenceClassification.from_pretrained(model_config.nn_model)
    model.to(model_config.device)

    predict(
        model, pos_loader, model_config, "../twitter-datasets/train_pos_full_irony.csv"
    )
    predict(
        model, neg_loader, model_config, "../twitter-datasets/train_neg_full_irony.csv"
    )
    predict(
        model,
        test_loader,
        model_config,
        "../twitter-datasets/test_data_irony.csv",
        include_ids=True,
    )


if __name__ == "__main__":
    main()
