#pylint:disable=no-member
import numpy as np
import cv2 as cv
import os

# Use built-in Haar cascade path
haar_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

# People list (labels must match training order)
people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']

# Load trained recognizer
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

# Load test image
img = cv.imread(r'C:/activity 8/opencv-course/Resources/Faces/eh.jpg')

# Check if image loaded
if img is None:
    raise FileNotFoundError("Image not found. Check your path!")

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Person', gray)

# Detect faces
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+w]

    # Predict
    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {people[label]} with a confidence of {confidence}')

    # Draw results
    cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow('Detected Face', img)
cv.waitKey(0)
cv.destroyAllWindows()
