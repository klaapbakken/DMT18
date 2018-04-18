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

def create_time_arr(user_df, m_tg=False):
    time = '23:59:59.999'
    if m_tg:
        n_user_df = user_df[user_df.variable == 'mood']
    else:
        n_user_df = user_df
    n = len(n_user_df.time.values)
    u_dates = np.unique(np.array([n_user_df.time.values[i][0:11] for i in range(n)]))
    time_arr = np.array([u_dates[i] + time for i in range(len(u_dates))])
    return time_arr

def manipulate_df_vals(df):
    org_df = df.copy()
    org_df = org_df.dropna(axis=0, how='any')
    val_df = org_df[org_df.variable == 'circumplex.valence']
    ar_df = org_df[org_df.variable == 'circumplex.arousal']

    org_df.loc[val_df.index.values, :].replace([-2, -1, 0, 1, 2], [1, 2, 3, 4, 5])
    org_df.loc[ar_df.index.values, :].replace([-2, -1, 0, 1, 2], [1, 2, 3, 4, 5])

    org_df[org_df.variable == 'circumplex.valence'] = org_df.loc[val_df.index.values, :].replace([-2, -1, 0, 1, 2],
                                                                                                 [1, 2, 3, 4, 5])
    org_df[org_df.variable == 'circumplex.arousal'] = org_df.loc[ar_df.index.values, :].replace([-2, -1, 0, 1, 2],
                                                                                                [1, 2, 3, 4, 5])
    return org_df.copy()

def extract_next_day_average(X, tg, df, m_pos):
    # Time point, the last one in a sequence as input. Create a timepoint at 23.59.999 on current day.
    # Locate all points in data frame with 0 < t_delta < 24*60*60 and variable corresponding to current.
    # Take the average, this is the response to current timepoint
    #Remove from X if next day average does not exist

    cols = X.shape[1]
    mn = '23:59:59.999'
    nda = []
    for i in range(cols):
        cur_day = tg[i][0:11] + mn
        next_day_indices = [j for j in range(len(tg)) if (0 <= t_delta(tg[j], cur_day) <= 24*60*60)
                            and df.variable.values[j] == 'mood']
        if not next_day_indices:
            X = np.delete(X, 0, i)
        else:
            nda.append(np.mean(X[m_pos, next_day_indices]))
    return X, nda

