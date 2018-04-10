import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime


def time_difference(t1, t2):
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


def time_earlier(t1, t2):
    year1 = int(t1[0:4])
    year2 = int(t2[0:4])
    if year1 == year2:
        month1 = int(t1[5:7])
        month2 = int(t2[5:7])
        if month1 == month2:
            day1 = int(t1[8:10])
            day2 = int(t2[8:10])
            if day1 == day2:
                hour1 = int(t1[11:13])
                hour2 = int(t2[11:13])
                if hour1 == hour2:
                    min1 = int(t1[14:16])
                    min2 = int(t2[14:16])
                    if min1 == min2:
                        sec1 = int(t1[17:19])
                        sec2 = int(t2[17:19])
                        if sec1 == sec2:
                            subsec1 = int(t1[20:23])
                            subsec2 = int(t2[20:23])
                            return subsec1 <= subsec2
                        else:
                            return sec1 < sec2
                    else:
                        return min1 < min2

                else:
                    return hour1 < hour2

            else:
                return day1 < day2
        else:
            return month1 < month2
    else:
        return year1 < year2


def create_time_series(df, umt):
    mood_times = df.time[df.variable == 'mood'].values
    mood_values = df.value[df.variable == 'mood'].values
    n = mood_values.shape[0]

    # unique_mt = np.unique(df.variable[df.variable != 'mood'].values).tolist()
    unique_mt = umt
    measurement_types = df.variable.values[df.variable != 'mood']
    measurement_values = df.value.values[df.variable != 'mood']
    measurement_times = df.time.values[df.variable != 'mood']
    l = measurement_times.shape[0]

    mean_types = np.array(['activity', 'circumplex.valence', 'circumplex.arousal'])
    sum_types = np.array(list(filter(lambda x: x not in mean_types, unique_mt)))

    X = np.zeros((len(unique_mt), n))
    indices_to_drop = np.array([])
    j = 0
    for m_time in mood_times:
        earlier_time_indices = np.array([k for k in range(l)
                                         if time_earlier(measurement_times[k], m_time)
                                         and k not in indices_to_drop])
        if len(earlier_time_indices) == 0:
            continue
        earlier_types = measurement_types[earlier_time_indices]
        i = 0
        for m_type in unique_mt:
            earlier_measurement_indices = np.nonzero(np.isin(earlier_types, m_type))[0]
            if len(earlier_measurement_indices) != 0:
                measurement_indices = earlier_time_indices[earlier_measurement_indices]
                if m_type in sum_types:
                    # Sum all measurement types that should be added up
                    X[i, j] = np.sum(measurement_values[measurement_indices])
                elif m_type in mean_types:
                    X[i, j] = np.mean(measurement_values[measurement_indices])
                indices_to_drop = np.concatenate((indices_to_drop, measurement_indices))
            i += 1
        j += 1

    y = mood_values

    return X, y


def shift_and_add_time(df, X, y):
    timestamps = df.time.values[df.variable == 'mood']
    t_delta = np.array([time_difference(timestamps[i], timestamps[i + 1]) for i in range(y.shape[0] - 1)])
    sX = X[:, :-1]
    sX = np.vstack((sX, t_delta))
    sy = y[1:]

    return sX, sy


def merge_user_data(df):
    umt = np.unique(df.variable.values[df.variable != 'mood'])
    ids = np.unique(df.id.values)
    n_ids = ids.shape[0]
    id_df_list = [df[df.id == ids[i]] for i in range(n_ids)]
    i = 0
    X, y = create_time_series(id_df_list[0], umt)
    X, y = shift_and_add_time(id_df_list[0], X, y)
    for i in range(1, n_ids):
        tX, ty = create_time_series(id_df_list[i], umt)
        tX, ty = shift_and_add_time(id_df_list[i], tX, ty)
        X = np.hstack((X, tX))
        y = np.hstack((y, ty))
        print(X.shape)
        print(y.shape)
    return X, y


def save_processed_to_csv(X, y, df):
    measurement_types = np.unique(df.variable[df.variable != 'mood'].values).tolist()
    cols = np.concatenate((measurement_types, np.array(['time', 'mood'])))
    data = np.vstack((X, y)).T
    proc_df = pd.DataFrame(data=data, columns=cols)
    proc_df.head()
    return proc_df.to_csv('full_processed_data.csv', index=False)


if __name__ == "__main__":
    df = pd.read_csv("./Data/dataset_mood_smartphone.csv")
    df = df.dropna(axis=0, how='any')
    ids = np.unique(df.id.values)
    user_df = df[df.id == ids[0]]

    X, y = merge_user_data(df)

    save_processed_to_csv(X, y, df)
