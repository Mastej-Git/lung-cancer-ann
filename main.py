# pylint: disable=import-error
# pylint: disable=missing-function-docstring
# pylint: disable=missing-module-docstring

import os
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers

from sklearn.model_selection import train_test_split
from sklearn import metrics

from PIL import Image
import cv2

PATH = "../lung_colon_image_set/lung_image_sets"
EPOCHS = 10
BATCH_SIZE = 64


def show_example_pics():
    catalogs = os.listdir(PATH)

    for catalog in catalogs:
        image_dir = f"{PATH}/{catalog}"
        images = os.listdir(image_dir)

        fig, ax = plt.subplots(1, 3)
        fig.suptitle(f"Images for {catalog} category: ")

        for i in range(3):
            k = np.random.randint(0, len(images))
            img = np.array(Image.open(f"{PATH}/{catalog}/{images[k]}"))
            ax[i].imshow(img)
            ax[i].axis("off")
        plt.show()


def prepare_data():
    IMG_SIZE = 256
    SPLIT = 0.2

    x, y = [], []

    catalogs = sorted(os.listdir(PATH))
    for i, catalog in enumerate(catalogs):
        images = glob(f"{PATH}/{catalog}/*.jpeg")

        for image in images:
            img = cv2.imread(image)

            x.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0)
            y.append(i)

    x = np.asarray(x)
    one_hot_encoded_y = pd.get_dummies(y).values

    x_train, x_val, y_train, y_val = train_test_split(
        x, one_hot_encoded_y, test_size=SPLIT, random_state=2022
    )
    return x_train, x_val, y_train, y_val

IMG_SIZE = 256
model = keras.models.Sequential([
    layers.Conv2D(filters=32,
                  kernel_size=(5, 5),
                  activation='relu',
                  input_shape=(IMG_SIZE,
                               IMG_SIZE,
                               3),
                  padding='same'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(filters=64,
                  kernel_size=(3, 3),
                  activation='relu',
                  padding='same'),

    layers.MaxPooling2D(2, 2),

    layers.Conv2D(filters=128,
                  kernel_size=(3, 3),
                  activation='relu',
                  padding='same'),

    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(3, activation='softmax')
])
model.summary()


model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

# prepare_data()
