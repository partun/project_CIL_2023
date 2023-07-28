import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, InputLayer, Input, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.linear_model import LinearRegression, LogisticRegression

np.random.seed(1)

MAX_SEQUENCE_LENGTH = 45
MAX_NB_WORDS = 10000
EMBEDDING_DIM = 50
VALIDATION_SPLIT = 0.1

GLOVE_MODEL_PATH = "../twitter-datasets/glove.6B.50d.txt"
D = 50 # each tweets represetned by N * D matrix
N = 45 # clip/pad each tweet length to 40
FULL = False
POS_FULL_PATH = "../twitter-datasets/train_pos_full.txt"
NEG_FULL_PATH = "../twitter-datasets/train_neg_full.txt"
POS_PART_PATH = "../twitter-datasets/train_pos.txt"
NEG_PART_PATH = "../twitter-datasets/train_neg.txt"


def read_glove_vector(glove_vec):
  """
  Return a map referring word to vector (np.array)
  """
  with open(glove_vec, 'r', encoding='UTF-8') as f:
    word_to_vec_map = {}
    for line in f:
      w_line = line.split()
      curr_word = w_line[0]
      word_to_vec_map[curr_word] = np.array(w_line[1:], dtype=np.float64)
  return word_to_vec_map

def load_tweets(filename, label, tweets, labels):
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            word_list = line.rstrip()
            tweets.append(word_list)
            labels.append(label)

def text_to_features(text):
    """
    Convert a tweet to its embedding, returns 40(fixed length) * dim array
    """
    features = []
    for word in text.split()[:MAX_SEQUENCE_LENGTH]:
        if word in glove_model:
            features.append(glove_model[word])
        else:
            features.append(np.zeros_like(glove_model["moon"]))

    while len(features) < MAX_SEQUENCE_LENGTH:
        features.append(np.zeros_like(glove_model["moon"]))

    return np.array(features)

def preprocess():
    """
    Load data, split into training and validation set
    """
    tweets = []
    labels = []
    load_tweets(POS_FULL_PATH, 1, tweets, labels)
    load_tweets(NEG_FULL_PATH, 0, tweets, labels)
  
    tweets = np.array(tweets)
    labels = np.array(labels)
    print("Number of tweets: " + str(len(tweets)))

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

def word_to_index(text):
    """
    takes tweets, return an array of word indices
    """
    indices = []
    for word in text.split()[:MAX_SEQUENCE_LENGTH]:
        if word in word_index:
            indices.append(word_index[word])
        else:
            indices.append(0)
    while len(indices) < MAX_SEQUENCE_LENGTH:
        indices.append(0)
    return np.array(indices)

class CNN():
    def __init__(self):
        pass

    def model(self, embedding_matrix):
        self.model = models.Sequential()
        self.model.add(Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], input_length=MAX_SEQUENCE_LENGTH, weights=[embedding_matrix], trainable=False))
        self.model.add(Conv1D(128, 3, activation='relu'))
        self.model.add(MaxPooling1D(pool_size=3))
        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.summary()

    def train(self, X_train, Y_train, X_val, Y_val):
        self.model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.BinaryAccuracy()])

        history = self.model.fit(X_train, Y_train, epochs=7, batch_size=128,
                    validation_split=0.1)
        metrics_df = pd.DataFrame(history.history)


# load tweets and GloVe embeddings
X_train, X_val, Y_train, Y_val = preprocess()
glove_model_path = GLOVE_MODEL_PATH
glove_model = read_glove_vector(glove_model_path)

# tokenize and pad
tokenizer = Tokenizer()
texts = X_train + X_val
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index # {word: index}

# create embedding matrix
embedding_matrix = np.zeros((len(word_index), D)) # {index: embedding}
for word, i in word_index.items():
  if i >= len(word_index):
    break
  embedding_vector = glove_model.get(word)
  if embedding_vector is not None:
    embedding_matrix[i] = embedding_vector
del glove_model

# turn tweet word list into index list
X_train_features = np.array([word_to_index(text) for text in X_train])
X_val_features = np.array([word_to_index(text) for text in X_val])
Y_train= np.asarray(Y_train)
Y_val= np.asarray(Y_val)

# train
cnn1 = CNN()
cnn1.model(embedding_matrix)
cnn1.train(X_train_features, Y_train, X_val_features, Y_val)