import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


def load_tweets(filename, label, tweets, labels):
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            word_list = line.rstrip().split(" ")
            word_list = " ".join(
                map(lambda x: "<hashtag>" if x.startswith("#") else x, word_list)
            )

            tweets.append(word_list)
            labels.append(label)


def preprocess():
    tweets = []
    labels = []

    load_tweets("../twitter-datasets/train_pos_full.txt", 1, tweets, labels)
    load_tweets("../twitter-datasets/train_neg_full.txt", 0, tweets, labels)

    tweets = np.array(tweets)
    labels = np.array(labels)

    print("Number of tweets: " + str(len(tweets)))

    np.random.seed(1)

    shuffle_indices = np.random.permutation(len(tweets))
    split_index = int(len(tweets) * 0.8)
    trian_indices = shuffle_indices[:split_index]
    val_indices = shuffle_indices[split_index:]

    print(f"Number of training tweets: {len(trian_indices)}")
    print(f"Number of validation tweets: {len(val_indices)}")

    vectorizer = CountVectorizer(max_features=20000, ngram_range=(1, 3))
    X_train = vectorizer.fit_transform(tweets[trian_indices])
    X_val = vectorizer.transform(tweets[val_indices])

    Y_train = labels[trian_indices]
    Y_val = labels[val_indices]

    model = LogisticRegression(C=1e5, max_iter=200)
    model.fit(X_train, Y_train)

    Y_train_pred = model.predict(X_train)
    Y_val_pred = model.predict(X_val)

    train_accuracy = (Y_train_pred == Y_train).mean()
    val_accuracy = (Y_val_pred == Y_val).mean()

    print(f"Accuracy (training set): {train_accuracy:.05f}")
    print(f"Accuracy (validation set): {val_accuracy:.05f}")


if __name__ == "__main__":
    print("Training bag of words model...")
    preprocess()
