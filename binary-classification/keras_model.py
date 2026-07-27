import os
os.environ["KERAS_BACKEND"] = "torch" 

import keras
import torch
from keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = 10000)

print(train_data)
print(train_data.shape)

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

model = keras.Sequential(
    keras.layers.Dense(units=32,activation="reLu"),
    keras.layers.Dense()
)