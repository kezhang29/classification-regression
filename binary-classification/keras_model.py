import os
os.environ["KERAS_BACKEND"] = "torch" 

import keras
from keras import layers
import torch
from keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = 10000)

print(train_data)
print(train_data.shape)
print(train_labels)
print(train_labels.shape)

# word_index is a dictionary mapping words to an integer index.
word_index = imdb.get_word_index()
# Reverses it, mapping integer indices to words
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# Decodes the review. Note that the indices are offset by 3 because 0,
# 1, and 2 are reserved indices for "padding," "start of sequence," and
# "unknown."
decoded_review = " ".join(
    [reverse_word_index.get(i - 3, "?") for i in train_data[0]]
)

print(decoded_review)

def multi_hot_encode(data, *,num_words):
    results = torch.zeros(size = (len(data),num_words))
    for i, sequence in enumerate(data):
        results[i][sequence] = 1
    return results

training_data = multi_hot_encode(train_data, num_words = 10000)
print(training_data[0])
test_data = multi_hot_encode(test_data, num_words = 10000)

train_labels = train_labels.astype("float32")
test_labels = test_labels.astype("float32")

x_train = training_data[10000:]
x_val = training_data[:10000]
y_train = test_labels[10000:]
y_val = test_labels[:10000]

model = keras.Sequential([
    layers.Dense(units=16,activation="relu"),
    layers.Dense(units=16,activation="relu"),
    layers.Dense(units=1,activation="sigmoid")]
)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(x_train,y_train,epochs=40,batch_size = 512,validation_data=(x_val,y_val))
