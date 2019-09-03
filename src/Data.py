import warnings

import pandas as pd
from keras.datasets import mnist

warnings.simplefilter("ignore")

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

X_train = pd.DataFrame(X_train)
y_train = pd.DataFrame(y_train)
X_test = pd.DataFrame(X_test)
y_test = pd.DataFrame(y_test)


X_train.to_csv("Data/X_train.csv", index=False)
y_train.to_csv("Data/y_train.csv", index=False)
X_test.to_csv("Data/X_test.csv", index=False)
y_test.to_csv("Data/y_test.csv", index=False)
