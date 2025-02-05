import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

dataset_path = './dataset'

img_width, img_height = 64, 64

def load_images(folder_path):
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        if img_path.endswith(".jpg") or img_path.endswith(".png"):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (img_width, img_height))
            img = img.astype('float32') / 255.0
            images.append(img)
            labels.append(1 if 'yes' in folder_path else 0)
    return images, labels

yes_images, yes_labels = load_images(os.path.join(dataset_path, 'yes'))
no_images, no_labels = load_images(os.path.join(dataset_path, 'no'))

images = yes_images + no_images
labels = yes_labels + no_labels

images = np.array(images)
labels = np.array(labels)

train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.3, random_state=42)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy')

model.fit(train_images, train_labels, epochs=20, batch_size=64)

model.save('brain_tumor_detection_model.h5')
