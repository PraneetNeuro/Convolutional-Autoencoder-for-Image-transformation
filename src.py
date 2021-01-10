import cv2
import tensorflow as tf
import os
import numpy as np
from tqdm import tqdm


class Dataset:
    def __init__(self, input_path, target_path):
        self.input_path = input_path
        self.target_path = target_path
        self.X = []
        self.Y = []
        self.make_synthetic_dataset()

    def make_synthetic_dataset(self):
        for img_name in tqdm(os.listdir(self.input_path)):
            img = cv2.imread(os.path.join(self.input_path, img_name))
            img = cv2.resize(img, (100, 100))
            img = np.array(img) / 255
            self.X.append(img)
        for img_name in tqdm(os.listdir(self.target_path)):
            img = cv2.imread(os.path.join(self.target_path, img_name))
            img = cv2.resize(img, (100, 100))
            img = np.array(img) / 255
            self.Y.append(img)
        assert len(self.X) == len(self.Y), "Err: src target mismatch. Dataset not clean"


class AutoEncoder:
    def __init__(self, dataset, epochs=10):
        self.model = tf.keras.Sequential()
        self.epochs = epochs
        self.dataset = dataset
        self.AutoEncoder()

    def AutoEncoder(self):
        self.model.add(tf.keras.layers.Input([100, 100, 3]))
        self.model.add(tf.keras.layers.Conv2D(64, (1, 1)))
        self.model.add(tf.keras.layers.LeakyReLU())
        self.model.add(tf.keras.layers.Conv2D(128, (1, 1)))
        self.model.add(tf.keras.layers.LeakyReLU())
        self.model.add(tf.keras.layers.Conv2D(256, (1, 1)))
        self.model.add(tf.keras.layers.LeakyReLU())
        self.model.add(tf.keras.layers.Conv2DTranspose(256, (1, 1), strides=(1, 1), activation='relu'))
        self.model.add(tf.keras.layers.Conv2DTranspose(128, (1, 1), strides=(1, 1), activation='relu'))
        self.model.add(tf.keras.layers.Conv2DTranspose(64, (1, 1), strides=(1, 1), activation='relu'))
        self.model.add(tf.keras.layers.Conv2DTranspose(3, (1, 1), strides=(1, 1), activation='sigmoid'))
        self.model.compile(tf.keras.optimizers.RMSprop(), tf.keras.losses.mean_squared_error, metrics=['accuracy'])
        self.model.summary()
        self.model.fit(x=np.asarray(self.dataset.X, dtype=np.float), y=np.asarray(self.dataset.Y, dtype=np.float),
                       epochs=self.epochs)
        self.model.save('model_{}'.format(self.epochs))
