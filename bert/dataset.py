import pandas as pd
import torch
from datasets import Features, ClassLabel, Value, Dataset, DatasetDict
from transformers import AutoTokenizer
import os


def load_dataset(frac=1, train_size=0.8, use_full_dataset=True):
    if use_full_dataset:
        neg_path = "../twitter-datasets/train_neg_full.txt"
        pos_path = "../twitter-datasets/train_pos_full.txt"
    else:
        neg_path = "../twitter-datasets/train_neg.txt"
        pos_path = "../twitter-datasets/train_pos.txt"

    tweets_neg = pd.read_csv(
        neg_path,
        sep="\t\t",
        lineterminator="\n",
        encoding="utf8",
        names=["tweet"],
    )

    tweets_pos = pd.read_csv(
        pos_path,
        sep="\t\t",
        lineterminator="\n",
        encoding="utf8",
        names=["tweet"],
    )

    tweets_neg["label"] = 0
    tweets_pos["label"] = 1
    tweets = pd.concat([tweets_neg, tweets_pos])

    tweets = tweets.sample(frac=frac).reset_index(drop=True)

    train_dataset = tweets.sample(frac=train_size, random_state=200)
    val_dataset = tweets.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    print("Dataset Head")
    print(train_dataset.head())

    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(
                train_dataset,
                features=Features(
                    {
                        "tweet": Value("string"),
                        "label": Value("float32")
                        # "label": ClassLabel(
                        #     names=["negative", "positive"],
                        #     names_file=None,
                        #     id=None,
                        # ),
                    }
                ),
            ),
            "validation": Dataset.from_pandas(
                val_dataset,
                features=Features(
                    {
                        "tweet": Value("string"),
                        "label": Value("float32")
                        # "label": ClassLabel(
                        #     names=["negative", "positive"],
                        #     names_file=None,
                        #     id=None,
                        # ),
                    }
                ),
            ),
        }
    )

    # if use_full_dataset:
    #     test_dataset = pd.read_csv(
    #         "../twitter-datasets/test_data.txt",
    #         sep=",",
    #         lineterminator="\n",
    #         encoding="utf8",
    #         names=["id", "tweet"],
    #     )
    #     dataset["test"] = (
    #         Dataset.from_pandas(
    #             test_dataset,
    #             features=Features(
    #                 {
    #                     "tweet": Value("string"),
    #                     "label": Value("float32")
    #                     # "label": ClassLabel(
    #                     #     names=["negative", "positive"],
    #                     #     names_file=None,
    #                     #     id=None,
    #                     # ),
    #                 }
    #             ),
    #         ),
    #     )

    dataset = dataset.with_format("torch")
    print("loaded dataset successfully!")
    print(dataset)
    return dataset


def get_preprocess(tokenizer):
    def preprocess(examples):
        input = tokenizer(
            examples["tweet"],
            truncation=True,
            max_length=45,
            padding="max_length",
            pad_to_max_length=True,
            return_token_type_ids=True,
        )

        return {
            "input_ids": torch.tensor(input["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(input["attention_mask"], dtype=torch.long),
            "token_type_ids": torch.tensor(input["token_type_ids"], dtype=torch.long),
            "label": examples["label"],
        }

    return preprocess


def tokenize_dataset(dataset, tokenizer_model):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)

    encoded_dataset = dataset.map(get_preprocess(tokenizer), batched=True)

    print("tokenized dataset successfully!")
    print(encoded_dataset)
    return encoded_dataset


def load_and_tokenize_dataset(
    model_config, frac=1, train_size=0.8, use_full_dataset=True
):
    if use_full_dataset:
        cache_path = "dataset_full_cache"
    else:
        cache_path = "dataset_cache"

    if os.path.exists(cache_path):
        print("Loading cached dataset...")
        dataset = DatasetDict.load_from_disk(cache_path)
        return dataset

    dataset = load_dataset(frac, train_size, use_full_dataset)
    dataset = tokenize_dataset(dataset, model_config.tokenizer_model)

    dataset.save_to_disk(cache_path)
    return dataset
