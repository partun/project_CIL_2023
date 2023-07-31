"""
Logistic regression model using bag of words features.

Running this script will fit the logistic regression model for all the different preprocessing strategies.
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


def load_tweets(filename, label, tweets, labels):
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            tweets.append(line)
            labels.append(label)


def get_dataset_path(dataset_type) -> tuple[str, str]:
    match dataset_type:
        case "full":
            neg_path = "../twitter-datasets/train_neg_full_notabs.csv"
            pos_path = "../twitter-datasets/train_pos_full_notabs.csv"
        case "small":
            neg_path = "../twitter-datasets/train_neg_notabs.csv"
            pos_path = "../twitter-datasets/train_pos_notabs.csv"
        case "noemoji":
            neg_path = "../twitter-datasets/train_neg_full_without_emoji.csv"
            pos_path = "../twitter-datasets/train_pos_full_without_emoji.csv"
        case "nostopwords":
            neg_path = "../twitter-datasets/train_neg_full_no_stopwords.csv"
            pos_path = "../twitter-datasets/train_pos_full_no_stopwords.csv"
        case "nopunctuation":
            neg_path = "../twitter-datasets/train_neg_full_no_punctuation.csv"
            pos_path = "../twitter-datasets/train_pos_full_no_punctuation.csv"
        case "split_hashtags":
            neg_path = "../twitter-datasets/train_neg_full_split_hashtags.csv"
            pos_path = "../twitter-datasets/train_pos_full_split_hashtags.csv"
        case "spellcheck":
            neg_path = "../twitter-datasets/train_neg_full_spellcheck.csv"
            pos_path = "../twitter-datasets/train_pos_full_spellcheck.csv"
        case "combined":
            neg_path = "../twitter-datasets/train_neg_full_combined.csv"
            pos_path = "../twitter-datasets/train_pos_full_combined.csv"
        case _:
            raise ValueError(f"Invalid dataset type: {dataset_type}")

    return pos_path, neg_path


def run(dataset_type):
    tweets = []
    labels = []

    pos_path, neg_path = get_dataset_path(dataset_type)
    load_tweets(pos_path, 1, tweets, labels)
    load_tweets(neg_path, 0, tweets, labels)

    tweets = np.array(tweets)
    labels = np.array(labels)

    print("Number of tweets: " + str(len(tweets)))

    np.random.seed(1)

    shuffle_indices = np.random.permutation(len(tweets))
    split_index = int(len(tweets) * 0.9)
    trian_indices = shuffle_indices[:split_index]
    val_indices = shuffle_indices[split_index:]

    print(f"Number of training tweets: {len(trian_indices)}")
    print(f"Number of validation tweets: {len(val_indices)}")

    vectorizer = CountVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(tweets[trian_indices])
    X_val = vectorizer.transform(tweets[val_indices])

    Y_train = labels[trian_indices]
    Y_val = labels[val_indices]

    model = LogisticRegression(C=1e5, max_iter=100)
    model.fit(X_train, Y_train)

    Y_train_pred = model.predict(X_train)
    Y_val_pred = model.predict(X_val)

    train_accuracy = (Y_train_pred == Y_train).mean()
    val_accuracy = (Y_val_pred == Y_val).mean()

    print(f"Accuracy (training set): {train_accuracy:.05f}")
    print(f"Accuracy (validation set): {val_accuracy:.05f}")


if __name__ == "__main__":
    print("Training bag of words model...")

    for dataset_type in ["full", "noemoji", "nostopwords", "nopunctuation", "spellcheck", "split_hashtags", "combined"]:
        print(f"Dataset type: {dataset_type}")
        run(dataset_type)
        print()
