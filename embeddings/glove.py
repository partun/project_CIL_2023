import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

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

    load_tweets("../twitter-datasets/train_pos_full.txt", 1, tweets, labels)
    load_tweets("../twitter-datasets/train_neg_full.txt", 0, tweets, labels)

    tweets = np.array(tweets)
    labels = np.array(labels)

    print("Number of tweets: " + str(len(tweets)))

    np.random.seed(1)

    shuffle_indices = np.random.permutation(len(tweets))
    split_index = int(len(tweets) * 0.8)
    train_indices = shuffle_indices[:split_index]
    val_indices = shuffle_indices[split_index:]

    print(f"Number of training tweets: {len(train_indices)}")
    print(f"Number of validation tweets: {len(val_indices)}")
    X_train = [tweets[i] for i in train_indices]
    X_val = [tweets[i] for i in val_indices]
    Y_train = [labels[i] for i in train_indices]
    Y_val = [labels[i] for i in val_indices]
    return X_train, X_val, Y_train, Y_val

def text_to_features(text):
    features = []
    for word in text.split():
        if word in glove_model:
            # print(word)
            # print(glove_model[word])
            features.append(glove_model[word])
    if len(features) > 0:
        return np.mean(features, axis=0)
    else:
        return np.zeros_like(glove_model["moon"]) # moon is a random word


if __name__ == "__main__":
    
    X_train, X_val, Y_train, Y_val = preprocess()
    # print(X_train, X_val, y_train, y_val)
    
    glove_model_path ="../glove.6B.50d.txt"  
    glove_model = read_glove_vector(glove_model_path)
    
    X_train_features = np.array([text_to_features(text) for text in X_train])   # <class 'numpy.ndarray'>
    X_val_features = np.array([text_to_features(text) for text in X_val])
    # print(X_train_features.shape)
    # print(type(X_train_features))
    model = LogisticRegression(C=1e5, max_iter=200)
    model.fit(X_train_features, Y_train)

    Y_train_pred = model.predict(X_train_features)
    Y_val_pred = model.predict(X_val_features)

    train_accuracy = (Y_train_pred == Y_train).mean()
    val_accuracy = (Y_val_pred == Y_val).mean()

    print(f"Accuracy (training set): {train_accuracy:.05f}")
    print(f"Accuracy (validation set): {val_accuracy:.05f}")