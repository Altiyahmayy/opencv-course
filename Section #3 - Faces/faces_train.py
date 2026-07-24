#pylint:disable=no-member
import os
import cv2 as cv
import numpy as np

# Make sure these names match EXACTLY with your folder names
people = ['Ben Afflek', 'Elton John', 'Jerry Seinfeld', 'Madonna', 'Mindy Kaling']

# Absolute path to training dataset
DIR = r'C:/activity 8/opencv-course/Media Files/Faces/train'

# Use built-in Haar cascade
haar_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        if not os.path.exists(path):
            print(f"Missing folder: {path}")
            continue

        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            img_array = cv.imread(img_path)

            if img_array is None:
                print(f"Could not read image: {img_path}")
                continue

            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()
print(f'Training done. Total samples: {len(features)}')

# Convert lists to numpy arrays
features = np.array(features, dtype='object')
labels = np.array(labels)

if len(features) == 0 or len(labels) == 0:
    raise ValueError("No training data found. Check your dataset folders and images.")

# Create and train recognizer
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(features, labels)

# Save model and data
face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)
