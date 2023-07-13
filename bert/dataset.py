import pandas as pd
import torch
from datasets import Features, ClassLabel, Value, Dataset, DatasetDict
from transformers import AutoTokenizer
import os


def load_dataset(dataset_type: str, frac=1, train_size=0.8):
    test_path = None
    match dataset_type:
        case "full":
            neg_path = "../twitter-datasets/train_neg_full_notabs.csv"
            pos_path = "../twitter-datasets/train_pos_full_notabs.csv"
            test_path = "../twitter-datasets/test_data_notabs.csv"
        case "small":
            neg_path = "../twitter-datasets/train_neg_notabs.csv"
            pos_path = "../twitter-datasets/train_pos_notabs.csv"
        case "noemoji":
            neg_path = "../twitter-datasets/train_neg_full_without_emoji.csv"
            pos_path = "../twitter-datasets/train_pos_full_without_emoji.csv"
            test_path = "../twitter-datasets/test_data_without_emoji.csv"
        case "nostopwords":
            neg_path = "../twitter-datasets/train_neg_full_no_stopwords.csv"
            pos_path = "../twitter-datasets/train_pos_full_no_stopwords.csv"
            test_path = "../twitter-datasets/test_data_no_stopwords.csv"
        case "nopunctuation":
            neg_path = "../twitter-datasets/train_neg_full_no_punctuation.csv"
            pos_path = "../twitter-datasets/train_pos_full_no_punctuation.csv"
            test_path = "../twitter-datasets/test_data_no_punctuation.csv"
        case "combined":
            neg_path = "../twitter-datasets/train_neg_full_combined.csv"
            pos_path = "../twitter-datasets/train_pos_full_combined.csv"
            test_path = "../twitter-datasets/test_data_combined.csv"
        case _:
            raise ValueError("Invalid dataset type")

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
                    {"tweet": Value("string"), "label": Value("float32")}
                ),
            ),
            "validation": Dataset.from_pandas(
                val_dataset,
                features=Features(
                    {"tweet": Value("string"), "label": Value("float32")}
                ),
            ),
        }
    )

    if test_path is not None:
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

        dataset["test"] = Dataset.from_pandas(
            test_dataset,
            features=Features({"tweet": Value("string"), "id": Value("int64")}),
        )

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

        input_ids = torch.tensor(input["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(input["attention_mask"], dtype=torch.long)
        token_type_ids = torch.tensor(input["token_type_ids"], dtype=torch.long)

        if "label" in examples:
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "label": examples["label"],
            }
        else:
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "id": examples["id"],
            }

    return preprocess


def tokenize_dataset(dataset, tokenizer_model):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)

    encoded_dataset = dataset.map(get_preprocess(tokenizer), batched=True)

    print("tokenized dataset successfully!")
    print(encoded_dataset)
    return encoded_dataset


def load_and_tokenize_dataset(model_config, frac=1, train_size=0.8, force_reload=False):
    cache_path = f"dataset_{model_config.dataset_type}_cache"

    if not force_reload and os.path.exists(cache_path):
        print("Loading cached dataset...")
        dataset = DatasetDict.load_from_disk(cache_path)
        print(dataset)
        return dataset

    dataset = load_dataset(model_config.dataset_type, frac, train_size)
    dataset = tokenize_dataset(dataset, model_config.tokenizer_model)

    dataset.save_to_disk(cache_path)
    return dataset
