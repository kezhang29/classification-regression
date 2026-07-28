import os
os.environ["KERAS_BACKEND"] = "torch" 

import keras_model
import keras

model = keras.models.load_model("models/binary_classification.keras")

print(keras_model.test_data[0])
print(keras_model.test_labels[0])
print(keras_model.test_data[0].shape)

prediction = model.predict(keras_model.test_data[0].reshape(1,10000))

print(prediction)

