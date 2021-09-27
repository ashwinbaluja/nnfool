import tensorflow as tf 
import numpy as np
from tensorflow.keras import layers
import pandas
import tensorflow_datasets as tfds
import cv2

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="checkpoints/mnist", save_freq='epoch', verbose=1)

idConv = {0: 0, 1: 217, 2: 482, 3: 491, 4: 497, 5: 566, 6: 569, 7: 571, 8: 574, 9: 701}
convertId = lambda x: idConv[x]
counter = 0
def resize(im, size):
    global counter
    counter += 1
    print(counter)
    #https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/
    old_size = im.shape[:2] # old_size is in (height, width) format

    ratio = float(size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format

    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = size - new_size[1]
    delta_h = size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)
    return new_im

ds = tfds.load(name="mnist", split="train")
ds = tfds.as_numpy(ds)
dsX = np.array([x['image'] for x in ds], dtype='float')
dsX /= 255
classifier = tf.keras.models.load_model("checkpoints/mnistclassifier")

print(classifier)
def lossFunc(targetIdx, classif, a):
    target = np.zeros(10)
    target[targetIdx] = 1
    def loss(y_true, y_pred):
        return a * tf.keras.losses.MSE(y_true, y_pred) + (1-a) * tf.keras.losses.categorical_crossentropy(target, classif(y_pred)[0])
    
    return loss

model = tf.keras.Sequential(layers=[
    layers.InputLayer(input_shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.MaxPool2D((2,2), padding="same"),
    layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
    layers.MaxPool2D((2,2), padding="same"),
    layers.Conv2D(8, (3, 3), activation="relu", padding="valid"), 
    layers.MaxPool2D((2,2), padding="same"),
    layers.Conv2DTranspose(8, (3, 3), activation="relu", padding="same"),
    layers.UpSampling2D((2, 2)),
    layers.Conv2DTranspose(16, (3, 3), activation="relu", padding="same"),
    layers.UpSampling2D((2, 2)),
    layers.Conv2DTranspose(32, (3, 3), activation="relu"),
    layers.UpSampling2D((2, 2)),
    layers.Conv2D(1, 3, activation='relu', padding='same'),

])


model.compile(optimizer='adam', loss=lossFunc(4, classifier, .995))
print(model.summary())

model.fit(x=dsX, y=dsX, callbacks=[cp_callback], epochs=1000)

