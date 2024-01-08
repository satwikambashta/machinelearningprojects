from tensorflow.keras.models import load_model
import cv2
import tensorflow as tf
import numpy as np
img = cv2.imread('sad.jpeg')
resize = tf.image.resize(img, (256,256))
new_model = load_model('my_model.h5')
yhat = new_model.predict(np.expand_dims(resize/255, 0))
if yhat > 0.8:
    print(f'Predicted class is Happy')
    cv2.putText(img, "happy", (10,170), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
else:
    print(f'Predicted class is Sad')
    cv2.putText(img, "sad", (10,170), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)