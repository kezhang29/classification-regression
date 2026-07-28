import os
os.environ["KERAS_BACKEND"] = "torch" 

import keras
from keras import layers
import torch
from keras.datasets import imdb
import matplotlib.pyplot as plt

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
y_train = train_labels[10000:]
y_val = train_labels[:10000]

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

def train():
    history = model.fit(x_train,y_train,epochs=4,batch_size = 512,validation_data=(x_val,y_val))
    model.save("models/binary_classification.keras")
    return history

def plot(history):
    history_dict = history.history
    training_loss = history_dict["loss"]
    val_loss = history_dict["val_loss"]
    training_accuracy = history_dict["accuracy"]
    val_accuracy = history_dict["accuracy"]

    fig, axs = plt.subplots(2,2,figsize=(10,10))
    axs[0][0].plot(training_loss)
    axs[0][0].set_title("Loss vs Epochs")
    axs[0][1].plot(val_loss)
    axs[0][1].set_title("Val Loss vs Epochs")
    axs[1][0].plot(training_accuracy)
    axs[1][0].set_title("Training Accuracy vs Epochs")
    axs[1][1].plot(val_accuracy)
    axs[1][1].set_title("Val Accuracy vs Epochs")

    plt.show()

if __name__ == "__main__":
    plot(train())
