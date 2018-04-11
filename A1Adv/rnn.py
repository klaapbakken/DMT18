import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, LSTM

X = np.load('X.npy')
y = np.load('y.npy')
tr_size = int(np.floor(X.shape[0]*0.7))
te_size = int(X.shape[0] - tr_size)

tr_X = X[:tr_size, :, :]
tr_y = y[:tr_size*1]
tr_y = tr_y[::1]

te_X = X[-te_size:, :, :]
te_y = y[-te_size*1:]
te_y = te_y[::1]

print(tr_X.shape, tr_y.shape)

model = Sequential()
model.add(LSTM(10, input_shape=(tr_X.shape[1], tr_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
history = model.fit(tr_X, tr_y, epochs=500, batch_size=50, validation_data=(te_X, te_y), shuffle=False)

yhat = model.predict(te_X).reshape((te_X.shape[0],))

rolled_y = np.roll(y, 1)
base_yhat =  rolled_y[-te_size*1:]
base_yhat = np.array(base_yhat[::1])

print(np.sum(np.abs(base_yhat - te_y))/len(yhat))
print(np.sum(np.abs(yhat - te_y))/len(yhat))

plt.plot(np.arange(len(yhat)), np.abs(base_yhat - te_y))
plt.plot(np.arange(len(yhat)), np.abs(yhat - te_y))

plt.show()
