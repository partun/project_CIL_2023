import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence



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

    load_tweets("../twitter-datasets/train_pos.txt", 1, tweets, labels)
    load_tweets("../twitter-datasets/train_neg.txt", 0, tweets, labels)

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


    Y_train = labels[trian_indices]
    Y_val = labels[val_indices]

    # code
    max_features = 2000
    nb_classes = 2
    batch_size = 16
    maxlen = 140

    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(tweets[trian_indices])
    sequences_train = tokenizer.texts_to_sequences(tweets[trian_indices])
    sequences_test = tokenizer.texts_to_sequences(tweets[val_indices])

    print('Pad sequences (samples x time)')
    X_train = sequence.pad_sequences(sequences_train, maxlen=maxlen)
    X_val = sequence.pad_sequences(sequences_test, maxlen=maxlen)

    Y_train = tf.keras.utils.to_categorical(Y_train, nb_classes)
    Y_val = tf.keras.utils.to_categorical(Y_val, nb_classes)

    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_val.shape)

    print('Build model...')
    model = Sequential()
    model.add(Embedding(max_features, 64))
    model.add(LSTM(64)) 
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    print('Train...')
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=1,
            validation_data=(X_val, Y_val))
    score, acc = model.evaluate(X_val, Y_val,
                                batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)


    print("Generating test predictions...")
    # preds = model.predict_classes(X_val, verbose=0)
    # code



if __name__ == "__main__":
    print("Training LSTM model...")
    preprocess()
