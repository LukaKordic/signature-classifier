import pickle

from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.models import Sequential

# LOAD PREVIOUSLY SAVED DATA
X = pickle.load(open('X.pickle', 'rb'))
y = pickle.load(open('y.pickle', 'rb'))

X = X / 255

model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(150, 150, 1)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(512, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X, y, batch_size=64, epochs=3, validation_split=0.1)
