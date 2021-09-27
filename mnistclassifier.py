import tensorflow as tf 
import numpy as np
from tensorflow.keras import layers
import pandas
import tensorflow_datasets as tfds
import cv2

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="checkpoints/mnistclassifier", save_freq='epoch', verbose=1)

dsTrain = tfds.load(name="mnist", split="train", as_supervised=True)
dsTrain = dsTrain.map(lambda x, y: (x/255, tf.one_hot(y, 10)))

dsTest = tfds.load(name="mnist", split="test", as_supervised=True)
dsTest = dsTest.map(lambda x, y: (x/255, tf.one_hot(y, 10)))

model = tf.keras.Sequential(layers=[
    layers.InputLayer(input_shape=(28, 28, 1)),
    layers.Conv2D(4, (3, 3), activation="relu", padding="same"),
    layers.MaxPool2D((2,2), padding="same"),
    layers.Conv2D(8, (3, 3), activation="relu", padding="same"),
    layers.MaxPool2D((2,2), padding="same"),
    layers.Conv2D(16, (3, 3), activation="relu", padding="valid"), 
    layers.MaxPool2D((2,2), padding="same"),
    layers.Conv2D(32, (3, 3), activation="relu", padding="valid"), 
    layers.MaxPool2D((2,2), padding="same"),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy())
print(model.summary())

model.fit(dsTrain.batch(32), epochs=1000, validation_data=dsTest.batch(16), batch_size=32, validation_batch_size=16, callbacks=[cp_callback])