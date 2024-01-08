import tensorflow as tf
import os
import cv2
import imghdr
import numpy as np
from matplotlib import pyplot as plt
from bing_image_downloader import downloader
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten #dropout

gpus = tf.config.experimental.list_physical_devices('GPU')

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
#already downloaded once so commented out
# downloader.download("happy people", limit=100, output_dir='images')
# downloader.download("indian person happy", limit=200, output_dir='images')
# downloader.download("sad people", limit=100, output_dir='images')
# downloader.download("indian person sad", limit=200, output_dir='images')

data_dir='images'

valid_extensions = ['jpeg', 'jpg', 'png']

os.listdir(data_dir)
for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            img = cv2.imread(image_path) #load to opencv
            tip = imghdr.what(image_path)
            if tip not in valid_extensions:
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e:
            print('Issue with image {}'.format(image_path))
            # os.remove(image_path)
data = tf.keras.utils.image_dataset_from_directory('images')
data_iterator = data.as_numpy_iterator()

batch = data_iterator.next()

fig, ax = plt.subplots(ncols=8, figsize=(20,20))
for idx, img in enumerate(batch[0][:8]):
    ax[idx].imshow(img.astype(int))
    # ax[idx].imshow(img)
    ax[idx].title.set_text(batch[1][idx])

#PREPROCESSING
    
#scale image values to 0 and 1
data = data.map(lambda x,y: (x/255, y))
# 255 is max RGB value
#apply map_func to each element of the dataset and return new dataset containing the transformed elements
data.as_numpy_iterator().next()


#split_data
train_size = int(len(data)*.7)
val_size = int(len(data)*.2)+1
test_size = int(len(data)*.1)+1
# test_size = int(len(data)*.1)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

#MODELLING

model = Sequential()
model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
# print(model.summary())

logdir='logs'

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])


#EVALUATE
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
pre = Precision()
re = Recall()
acc = BinaryAccuracy()
for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)
print(pre.result(), re.result(), acc.result())

import cv2
import time

cap=cv2.VideoCapture(0)
pTime=cTime=0
while True:
    success, img = cap.read()
    resize = tf.image.resize(img, (256,256))
    yhat = model.predict(np.expand_dims(resize/255, 0))
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

from keras.models import load_model
#SAVE THE MODEL
model.save('my_model.h5')
