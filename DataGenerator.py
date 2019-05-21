from __future__ import absolute_import, division, print_function, unicode_literals

import os
import pickle
import random

import cv2
import numpy as np

# LOAD IMAGES FROM DISK
DATA_DIRECTORY = "../../Documents/signatures"
CATEGORIES = ['original', 'fake']

for category in CATEGORIES:
    path = os.path.join(DATA_DIRECTORY, category)
    for image in os.listdir(path):
        image_array = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE)

# CREATE TRAINING DATASET
training_data = []
IMG_SIZE = 150


def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATA_DIRECTORY, category)
        # MAP ORIGINAL AND FAKE CATEGORIES TO A NUMBER THAT WILL REPRESENT IT'S PREDICTION CLASS
        class_number = CATEGORIES.index(category)
        for image in os.listdir(path):
            image_array = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE)
            resized_image_array = cv2.resize(image_array, (IMG_SIZE, IMG_SIZE))
            training_data.append([resized_image_array, class_number])


create_training_data()
random.shuffle(training_data)
print(len(training_data))

X = []
y = []
for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# SAVE DATA SO WE DON'T HAVE TO DO PREVIOUS WORK EVERY TIME BEFORE TWEAKING NEURAL NET PARAMETERS
pickle_out = open('X.pickle', 'wb')
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open('y.pickle', 'wb')
pickle.dump(y, pickle_out)
pickle_out.close()
