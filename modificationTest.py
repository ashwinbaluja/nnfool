import tensorflow as tf 
import numpy as np
from tensorflow.keras import layers
import pandas
import tensorflow_datasets as tfds
import cv2
import tensorflow.keras.backend as K


cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="checkpoints/modification", save_freq='epoch', verbose=1)

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

ds = tfds.load(name="imagenette/320px-v2", split="validation")
ds = tfds.as_numpy(ds)
dsX = np.array([resize(x['image'], 224) / 255 for x in ds])
dsY = np.array([np.zeros((224, 224, 3)) for x in range(len(dsX))])
classifier = tf.keras.applications.VGG19()
print(classifier)

def lossFunc(targetIdx, vgg, a):
    target = np.zeros(1000)
    target[targetIdx] = 1
    targetTensor = K.constant(target)
    targetImage = K.constant(np.zeros((224, 224, 3)))
    mseFunc = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
    def loss(y_true, y_pred):
        mse = mseFunc(targetImage, y_pred)
        print(mse)
        return mse/(255 * 255 * 3) + tf.keras.losses.categorical_crossentropy(targetTensor, vgg(tf.keras.applications.vgg19.preprocess_input(tf.multiply(y_pred, y_true)))[0])
    
    return loss

model = tf.keras.Sequential(layers=[
    layers.InputLayer(input_shape=(224, 224, 3)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPool2D(),
    layers.Conv2D(16, (3, 3), activation="relu"),
    layers.MaxPool2D(),
    layers.Conv2D(8, (3, 3), activation="relu"), 
    
    layers.Conv2DTranspose(8, (3, 3), activation="relu"),
    layers.UpSampling2D(),
    layers.Conv2DTranspose(16, (3, 3), activation="relu"),
    layers.UpSampling2D(),
    layers.Conv2DTranspose(32, (3, 3), activation="relu"),
    layers.Conv2DTranspose(3, kernel_size=3, activation='relu')
])


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.001), loss=lossFunc(417, classifier, .005))
print(model.summary())

model.fit(x=dsX, y=dsX, batch_size=8, callbacks=[cp_callback], epochs=1000)

