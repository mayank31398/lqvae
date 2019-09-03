import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

plt.switch_backend("QT5Agg")


def LoadData(data_path, normalize=True, make_onehot=True, file="csv"):
    if file == "csv":
        X_train = pd.read_csv(os.path.join(data_path, "X_train.csv")).values
        y_train = pd.read_csv(os.path.join(
            data_path, "y_train.csv")).values.reshape(-1)
        X_test = pd.read_csv(os.path.join(data_path, "X_test.csv")).values
        y_test = pd.read_csv(os.path.join(
            data_path, "y_test.csv")).values.reshape(-1)

        X_train = X_train.reshape(-1, 28, 28, 1)
        X_test = X_test.reshape(-1, 28, 28, 1)

        if normalize:
            X_train = (X_train / 255) * 2 - 1
            X_test = (X_test / 255) * 2 - 1

        if make_onehot:
            one_hot = OneHotEncoder()
            y_train = one_hot.fit_transform(
                np.expand_dims(y_train, axis=1)).toarray()
            y_test = one_hot.transform(
                np.expand_dims(y_test, axis=1)).toarray()

        return (X_train, y_train), (X_test, y_test)
    elif file == "npy":
        X = np.load(os.path.join(data_path, "X_test_FGSM.npy"))
        y = np.load(os.path.join(data_path, "y_test_FGSM.npy"))

        for i in range(X.shape[0]):
            m = np.min(X[i])
            X[i] = X[i] - m
            M = np.max(X[i])
            X[i] = X[i] / M
            X[i] = 2 * X[i] - 1

        return X, y


def SaveImages(images, save_path, name="image"):
    os.makedirs(save_path, exist_ok=True)
    batch_size = images.shape[0]

    size = int(np.sqrt(batch_size))
    image_shape = images.shape[1]

    plt.figure(figsize=(size, size))

    for i in range(batch_size):
        plt.subplot(size, size, i+1)
        image = images[i, ...]
        image = np.reshape(image, [image_shape, image_shape])
        plt.imshow(image, cmap='gray')
        plt.axis('off')

    plt.subplots_adjust(
        top=0.976,
        bottom=0.049,
        left=0.01,
        right=0.978,
        hspace=0.01,
        wspace=0.01
    )

    plt.savefig(os.path.join(save_path, name + ".png"))
