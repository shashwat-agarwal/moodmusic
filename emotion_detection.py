#!/usr/bin/env python
# coding: utf-8

# In[1]:

#importing required libraries
import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.models import load_model
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import tkinter as tk
from tkinter import *

#loading pretrained model
model = load_model('model.h5')

#interacting with camera for face detection
cam = cv2.VideoCapture(0)
cv2.namedWindow("Image Capturing Started....")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    ret, frame = cam.read()

    if not ret:
        print("Framing Failed!")
        break

    # converting image from RGB to GRAYSCALE
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)

    # Forming a box around the face being detected
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow("Detecting face (Press Spacebar to capture):", frame)

    # Detecting which key is pressed
    k = cv2.waitKey(1)
    if k % 256 == 32:
        # PRESS SPACEBAR TO MOVE FORWARD
        img_name = "face.jpg"
        cv2.imwrite(img_name, frame)
        print("\n\nImage Captured! \n\nDetecting emotion/mood.....\n")
        # x = image.load_img('face.jpg')
        # plt.imshow(x)
        break

cam.release()
cv2.destroyAllWindows()


img = cv2.imread('face.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 4)

# Cropping out the face from the whole captured image
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), 2)
    faces = img[y:y + h, x:x + w]
    cv2.imwrite('face.jpg', faces)

cv2.waitKey(20)

# Converting image to array for emotion prediction
img = image.load_img('face.jpg', color_mode='rgb', target_size=(48, 48))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

# Image range is kept between 0-255
x /= 255

print(x.shape)

# Deleting image from system
os.remove("face.jpg")

top = tk.Tk()
top.title("Mood based Music Recommendation System")

list_of_songs=tk.Text(top,height=20,width=60)

# Predicting emotion/mood 
result = model.predict(x)
# print(result)
result = list(result[0])
print(result)

# Mapping result values to mood labels
label_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}
emotion_index = result.index(max(result))
emotion_result = label_dict[emotion_index]
score = round(max(result) * 100, 1)

# Printing detected emotion along with accuracy of detection
print("Detected emotion/mood: {} \nAccuracy: {}%".format(emotion_result, score))
detected_mood="Detected emotion/mood: {} \nAccuracy: {}%".format(emotion_result, score)
list_of_songs.insert(tk.INSERT,detected_mood)

# Plotting graph for accuracy vs emotion label
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
factor = result
ax.bar(emotions, factor)

# Displaying the graph and automatically closing the window after 5 seconds
plt.show(block=False)
plt.pause(5)
plt.close()


var="\nFetching songs for "+ emotion_result+ " mood.....\n\n";
list_of_songs.insert(tk.INSERT,var)


# Fetching songs corresponding to detected emotion from FireBase Realtime Database
cred = credentials.Certificate('music.json')

firebase_admin.initialize_app(cred, {
    'databaseURL': "https://mood-music-db-default-rtdb.firebaseio.com/"
})

# Querying the database to get songs of detected emotion
res = "/" + emotion_result
ref = db.reference(res)
snapshot = ref.get()

list_of_songs.insert(tk.INSERT,"Songs fetched!\n\n")

for key, val in snapshot.items():
    list_of_songs.insert(tk.INSERT,val+"\n")

list_of_songs.insert(tk.INSERT,"\n\n-------end--------\n")

list_of_songs.config(state="disabled")
list_of_songs.pack()
top.mainloop()
