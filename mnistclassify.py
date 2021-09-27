import tensorflow as tf 
import numpy as np
from tensorflow.keras import layers
import pandas
import tensorflow_datasets as tfds
import cv2

reconstructed_model = tf.keras.models.load_model("checkpoints/mnistclassifier")

dsTest = tfds.load(name="mnist", split="test", as_supervised=True)
dsTest = dsTest.map(lambda x, y: (x/255, tf.one_hot(y, 10)))

dsTestNp = tfds.as_numpy(dsTest)

a = None
for i in dsTestNp:
    a = i 
    break

print(a)

print(reconstructed_model(a[0].reshape(1, 28, 28, 1)))