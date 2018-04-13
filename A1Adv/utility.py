import datetime
import numpy as np


def t_delta(t1, t2):
    """Returns the difference in time between t1 and t2, expressed as seconds.
     Negative values indicate that t2 occurs before t1

    Arguments:
    t1 -- string containing a timestamp, as seen in  dataset_mood_smartphone.csv
    t2 -- same as t1
    """
    year1 = int(t1[0:4])
    year2 = int(t2[0:4])
    month1 = int(t1[5:7])
    month2 = int(t2[5:7])
    day1 = int(t1[8:10])
    day2 = int(t2[8:10])
    hour1 = int(t1[11:13])
    hour2 = int(t2[11:13])
    min1 = int(t1[14:16])
    min2 = int(t2[14:16])
    sec1 = int(t1[17:19])
    sec2 = int(t2[17:19])
    subsec1 = int(t1[20:23])
    subsec2 = int(t2[20:23])
    datetime1 = datetime.datetime(year1, month1, day1, hour1, min1, sec1, subsec1 * 1000)
    datetime2 = datetime.datetime(year2, month2, day2, hour2, min2, sec2, subsec2 * 1000)

    return (datetime2 - datetime1).total_seconds()


def rnn_reshape(X, y, l):
    # Reshaping
    m = X.shape[0]
    n = X.shape[1]
    cut = n % l
    rX = X[:, cut:].reshape(n // l, l, m)
    ry = y[cut:]

    return rX, ry


def rnn_reshape_2(X, y, l, rb):
    eX = np.empty((X.shape[0]*l//rb, X.shape[1], X.shape[2]))
    #1000 8 14
    fX = X.reshape((X.shape[0]*l, X.shape[2]))
    #1000 14
    for i in range(eX.shape[1]):
        eX[i, :, :] = fX[:l, :]
        fX = np.roll(fX, -rb, axis = 0)
    ey = y
    return eX, ey
