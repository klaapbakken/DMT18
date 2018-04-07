import numpy as np
import pandas as pd

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


def create_time_series(df):
    mood_times = df.time[df.variable == 'mood'].values
    mood_values = df.value[df.variable == 'mood'].values
    n = mood_values.shape[0]

    unique_mt = np.unique(df.variable[df.variable != ' mood'].values).tolist()
    measurement_types = df.variable.values[df.variable != 'mood']
    measurement_values = df.value.values[df.variable != 'mood']
    measurement_times = df.time.values[df.variable != 'mood']
    l = measurement_times.shape[0]

    mean_types = np.array(['activity', 'circumplex.valence', 'circumplex.arousal'])
    sum_types = np.array(list(filter(lambda x: x not in mean_types, unique_mt)))

    X = np.zeros((len(unique_mt), n))
    indices_to_drop = np.array([])
    for j in mood_times:
        earlier_time_indices = np.array([k for k in range(l) if time_earlier(measurement_times[k], j)])
        earlier_types = measurement_types[earlier_time_indices]
        for m in unique_mt:
            i = 0
            earlier_measurement_indices = np.nonzero(np.isin(earlier_types, m))[0]
            if len(earlier_measurement_indices) != 0:
                if m in sum_types:
                    # Sum all measurement types that should be added up
                    X[i, j] = np.sum(measurement_values[earlier_measurement_indices])
                elif m in mean_types:
                    X[i, j] = np.mean(measurement_values[earlier_measurement_indices])
                i += 1
                np.concatenate(indices_to_drop, earlier_measurement_indices)
            measurement_times = np.delete(measurement_times, indices_to_drop)
            l -= len()

    y = mood_values

    return X, y

df = pd.read_csv("./Data/dataset_mood_smartphone.csv")
df = df.dropna(axis=0, how='any')
ids = np.unique(df.id.values)
user_df = df[df.id == ids[0]]
X, y = create_time_series(user_df)
