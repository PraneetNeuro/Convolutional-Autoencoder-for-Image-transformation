import cv2
import tensorflow as tf
import os
import numpy as np
from tqdm import tqdm


class Dataset:
    def __init__(self, input_path=None, target_path=None, save=False, load_x=None, load_y=None):
        self.input_path = input_path
        self.target_path = target_path
        self.X = []
        self.Y = []
        self.save = save
        self.x_path = load_x
        self.y_path = load_y
        self.make_synthetic_dataset()

    def make_synthetic_dataset(self):
        if self.x_path is None and self.y_path is None and self.input_path is not None and self.target_path is not None:
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
            if self.save:
                np.save('x-dataset', self.X)
                np.save('y-dataset', self.Y)
        elif self.x_path is not None and self.y_path is not None:
            self.X = np.load(self.x_path)
            self.Y = np.load(self.y_path)


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
