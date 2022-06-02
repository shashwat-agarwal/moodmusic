################################ MUSIC RECOMMENDATION SYSTEM BASED ON EMOTION DETECTION ################################

from zipfile import ZipFile
file_name = "archive.zip"
with ZipFile(file_name, 'r') as zip:
zip.extractall()
print("Done")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import cv2


train_img_datagen = ImageDataGenerator(rescale=1./255)
val_img_datagen = ImageDataGenerator(rescale=1./255)


train_ds = train_img_datagen.flow_from_directory(
'train',
target_size = (48, 48),
batch_size = 64,
color_mode = 'rgb',
class_mode='categorical')

val_ds = val_img_datagen.flow_from_directory(
'test',
target_size = (48, 48),
batch_size = 64,
color_mode = 'rgb',
class_mode='categorical')

print(val_ds.class_indices)


base_model = tf.keras.applications.ResNet50(input_shape=(48,48,3),include_top=False, weights='imagenet')

model = Sequential()
model.add(base_model)
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', padding='same', input_shape=(48,48,3), data_format='channels_last'))
model.add(Conv2D(64, kernel_size=(3,3),padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=(3,3),padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
model.add(Conv2D(128, kernel_size=(3,3),padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024,activation='relu'))
model.add(Dropout(0.50))
model.add(Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6, amsgrad='True'), metrics=['accuracy'])

epochs = 12
model_info=model.fit_generator(train_ds, steps_per_epoch=449, epochs=epochs, validation_data=val_ds, validation_steps=112)

model.save('model.h5')

acc = model_info.history['accuracy']
val_acc = model_info.history['val_accuracy']
loss = model_info.history['loss']
val_loss = model_info.history['val_loss']
epochs_range = range(epochs)
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()