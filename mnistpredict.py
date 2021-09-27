import tensorflow as tf
import numpy as np
import cv2
import tensorflow_datasets as tfds

def lossFunc(targetIdx, vgg, a):
    target = np.zeros(1000)
    target[targetIdx] = 1
    def loss(y_true, y_pred):
        return a * tf.keras.losses.MSE(y_true, y_pred)# + (1-a) * tf.keras.losses.categorical_crossentropy(target, vgg(tf.keras.applications.vgg19.preprocess_input(y_pred))[0])
    
    return loss


classifier = None

reconstructed_model = tf.keras.models.load_model("checkpoints/mnist", custom_objects={'loss': lossFunc(417, classifier, 1)})

def resize(im, size):

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

index = 10
    
cv2.imshow('test', dsX[index] * 255)

im = reconstructed_model.predict(dsX[index].reshape(1, 28, 28 ,1))

reconstructed_model2 = tf.keras.models.load_model("checkpoints/mnistclassifier")

result2 = reconstructed_model2(im.reshape(1, 28, 28, 1))


cv2.imshow('result', im[0] * 255)
print(result2)
cv2.waitKey(0)2