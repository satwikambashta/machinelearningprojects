from tensorflow.keras.models import load_model
import cv2
import tensorflow as tf
import numpy as np
import cv2
import time

new_model = load_model('./models/videoClassifier.h5')

cap=cv2.VideoCapture(0)
pTime=cTime=0

while True:
    success, img = cap.read()
    resize = tf.image.resize(img, (256,256))
    yhat = new_model.predict(np.expand_dims(resize/255, 0))
    if yhat > 0.8:
        print(f'Predicted class is Happy')
        cv2.putText(img, "happy", (10,170), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
    else:
        print(f'Predicted class is Sad')
        cv2.putText(img, "sad", (10,170), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

    cv2.imshow("Image", img)
  
    key = cv2.waitKey(1)
    if key == ord('q'):
        break