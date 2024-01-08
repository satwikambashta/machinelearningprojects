from tensorflow.keras.models import load_model
import cv2
import tensorflow as tf
import numpy as np
img = cv2.imread('sad.jpeg')
resize = tf.image.resize(img, (256,256))
new_model = load_model('imageclassifier.h5')
new_model.predict(np.expand_dims(resize/255, 0))