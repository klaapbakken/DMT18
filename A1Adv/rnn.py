import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation, BatchNormalization, GRU
from keras import optimizers, regularizers
from keras import callbacks


def rnn(X, y):

    model = Sequential()
    model.add(BatchNormalization(input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(32))
    model.add(LSTM(32, input_shape=(X.shape[1], X.shape[2]), kernel_regularizer=regularizers.l2()))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    model.fit(X, y, epochs=300, batch_size=64, shuffle=False)

    return model


def fcnn(X, y):
    model = Sequential()
    model.add(Dense(1024, input_shape=(X.shape[0], X.shape[1])))
    model.add(Dropout(0.9))
    model.add(Dense(64))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    model.fit(X, y, epochs=500, batch_size=32, shuffle=False)

    return model
