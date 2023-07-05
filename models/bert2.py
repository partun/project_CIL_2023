import transformers
import pandas as pd
import datasets
import random

tweets_neg = pd.read_csv(
    "twitter-datasets/train_neg.txt",
    sep="\t",
    lineterminator="\n",
    encoding="utf8",
    names=["tweet"],
)
tweets_pos = pd.read_csv(
    "twitter-datasets/train_pos.txt",
    sep="\t",
    lineterminator="\n",
    encoding="utf8",
    names=["tweet"],
)

tweets_neg["label"] = 0
tweets_pos["label"] = 1
tweets = pd.concat([tweets_neg, tweets_pos])

tweets = tweets.sample(frac=0.5).reset_index(drop=True)

print(f"Total tweets: {len(tweets)}")

print(tweets.head())


def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(
        dataset
    ), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset) - 1)
        while pick in picks:
            pick = random.randint(0, len(dataset) - 1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    print(df)
    for column, typ in dataset.features.items():
        if isinstance(typ, datasets.ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])


from transformers import AutoTokenizer

model_checkpoint = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


t = tokenizer("Hello, this is a sentence!")



def preprocess(examples):
    return tokenizer(examples["tweet"], truncation=True)

print(preprocess(tweets.head()))
