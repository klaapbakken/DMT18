import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation


def rnn(X, y):
    model = Sequential()
    model.add(Dense(48, input_shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(16, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(1))
    model.add(Activation('linear'))
    model.compile(loss='mae', optimizer='adam')
    model.fit(X, y, epochs=100, batch_size=32, shuffle=False)

    return model


def fcnn(X, y):
    model = Sequential()
    model.add(Dense(128, input_shape=(X.shape[0], X.shape[1])))
    model.add(Dropout(0.9))
    model.add(Dense(64))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    model.fit(X, y, epochs=200, batch_size=32, shuffle=False)

    return model
