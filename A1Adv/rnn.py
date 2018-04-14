import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout


def train_model(X, y, seq_shift):
    tr_y = y[::seq_shift]

    model = Sequential()
    model.add(Dense(512, input_shape=(tr_X.shape[1], tr_X.shape[2])))
    model.add(Dropout(0.5))
    model.add(Dense(256))
    model.add(LSTM(32, input_shape=(tr_X.shape[1], tr_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    history = model.fit(tr_X, tr_y, epochs=100, batch_size=32, shuffle=False)

    return model


if __name__ == "__main__":
    X = np.load('X.npy')
    y = np.load('y.npy')
    print(X.shape, y.shape)
    tr_size = int(np.floor(X.shape[0] * 0.7))
    te_size = int(X.shape[0] - tr_size)

    tr_X = X[:tr_size, :, :]
    tr_y = y[:tr_size * 2]
    tr_y = tr_y[::2]

    te_X = X[-te_size:, :, :]
    te_y = y[-te_size * 2:]
    te_y = te_y[::2]

    print(tr_X.shape, tr_y.shape)

    model = Sequential()
    model.add(Dense(512, input_shape=(tr_X.shape[1], tr_X.shape[2])))
    model.add(Dropout(0.5))
    model.add(Dense(256))
    model.add(LSTM(32, input_shape=(tr_X.shape[1], tr_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    history = model.fit(tr_X, tr_y, epochs=200, batch_size=12, validation_data=(te_X, te_y), shuffle=False)

    yhat = model.predict(te_X).reshape((te_X.shape[0],))

    rolled_y = np.roll(y, 2)
    base_yhat = rolled_y[-te_size * 2:]
    base_yhat = np.array(base_yhat[::2])

    print(np.sum(np.abs(base_yhat - te_y)) / len(yhat))
    print(np.sum(np.abs(yhat - te_y)) / len(yhat))

    plt.plot(np.arange(len(yhat)), np.abs(base_yhat - te_y))
    plt.plot(np.arange(len(yhat)), np.abs(yhat - te_y))

    plt.show()
