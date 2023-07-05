import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
import tensorflow_hub as hub
import tensorflow as tf
#Elmo
elmo = hub.load("https://tfhub.dev/google/elmo/2").signatures["default"]

def read_glove_vector(glove_vec):
  with open(glove_vec, 'r', encoding='UTF-8') as f:
    words = set()
    word_to_vec_map = {}
    for line in f:
      w_line = line.split()
      curr_word = w_line[0]
      word_to_vec_map[curr_word] = np.array(w_line[1:], dtype=np.float64)
  return word_to_vec_map

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

    load_tweets("twitter-datasets/train_pos.txt", 1, tweets, labels)
    load_tweets("twitter-datasets/train_neg.txt", 0, tweets, labels)

    tweets = np.array(tweets)
    labels = np.array(labels)

    print("Number of tweets: " + str(len(tweets)))

    np.random.seed(1)

    shuffle_indices = np.random.permutation(len(tweets))
    split_index = int(len(tweets) * 0.8)
    train_indices = shuffle_indices[:split_index][:100]
    val_indices = shuffle_indices[split_index:][:20]

    print(f"Number of training tweets: {len(train_indices)}")
    print(f"Number of validation tweets: {len(val_indices)}")
    X_train = [tweets[i] for i in train_indices]
    X_val = [tweets[i] for i in val_indices]
    Y_train = [labels[i] for i in train_indices]
    Y_val = [labels[i] for i in val_indices]
    return X_train, X_val, Y_train, Y_val

# def text_to_features(text):
#     features = []
#     for word in text.split():
#         if word in glove_model:
#             # print(word)
#             # print(glove_model[word])
#             features.append(glove_model[word])
#     if len(features) > 0:
#         return np.mean(features, axis=0)
#     else:
#         return np.zeros_like(glove_model["moon"]) # moon is a random word


if __name__ == "__main__":
    
    X_train, X_val, Y_train, Y_val = preprocess()
    print(Y_train)
    # print(X_train, X_val, y_train, y_val)
    X_train_features = elmo(tf.constant(X_train))["elmo"] # , signature="default", as_dict=True
    X_val_features = elmo(tf.constant(X_val))["elmo"] # , signature="default", as_dict=True
    print(X_train_features.ndim)
    