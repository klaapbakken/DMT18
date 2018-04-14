import numpy as np
import pandas as pd

from utility import t_delta, rnn_reshape, rnn_reshape_2, create_time_arr


def create_time_series(user_df, u_vars, time_arr=None):
    """Creates a matrix containing information on activity between specific time points

    user_df - the data frame corresponding to a specific user, as seen when importing data from
    dataset_mood_smartphone.csv
    u_vars - numpy array containing information on variables that should be considered
    time_arr - numpy array containing time points (default = None). If None, then time points of mood measurement
    is used
    """

    # Getting time grid
    if time_arr is None:
        assert 'mood' not in u_vars
        # Data frame with mood rows removed
        nm_user_df = user_df[user_df.variable == 'mood']
        tg = nm_user_df.time.values

    else:
        tg = time_arr

    # Values, times and variables corresponding to variable list
    var_df = user_df[user_df.variable.isin(u_vars)]
    var_vals = var_df.value.values
    var_times = var_df.time.values
    var_arr = var_df.variable.values
    # Number of variables values, and number of time points
    n = tg.shape[0]
    l = var_times.shape[0]

    #Variables to sum, variables to average
    m_vars = np.array(['activity', 'circumplex.valence', 'circumplex.arousal', 'mood'])
    s_vars = np.array(list(filter(lambda x: x not in m_vars, u_vars)))

    #Covariate matrix
    X = np.zeros((len(u_vars), n))
    #Indices of values that should not be included next loop
    drop_list = np.array([])

    j = 0
    #Iteration through time array
    for t in tg:
        #Indices of time points preceding current time point
        prev_t_i = np.array([k for k in range(l)
                             if t_delta(var_times[k], t) >= 0
                             and k not in drop_list])
        if len(prev_t_i) == 0:
            continue
        #Variables that correspond to earlier time points
        prev_vars = var_arr[prev_t_i]
        i = 0
        for u_var in u_vars:
            #Indices that correspond to values that should be included in design matrix this loop
            prev_m_i = np.nonzero(np.isin(prev_vars, u_var))[0]
            if len(prev_m_i) != 0:
                #Indices in array for all values
                m_i = prev_t_i[prev_m_i]
                #To sum
                if u_var in s_vars:
                    X[i, j] = np.sum(var_vals[m_i])
                #To average
                elif u_var in m_vars:
                    X[i, j] = np.mean(var_vals[m_i])
                drop_list = np.concatenate((drop_list, m_i))
            i += 1
        j += 1

    return X


def shift_and_add_time(df, X, y, skip_time=True):
    #Adding time since last mood measurement as feature
    #Shifting response one position to the right
    if not skip_time:
        time_arr = df.time.values[df.variable == 'mood']
        t_d = np.array([t_delta(time_arr[i], time_arr[i + 1]) for i in range(y.shape[0] - 1)])
    sX = X[:, :-1]
    if not skip_time:
        sX = np.vstack((sX, t_d))
    sy = y[1:]

    return sX, sy


def merge_user_data(df, reshape, mean=False):
    #Collecting data from all users in a data frame into a feature matrix
    #Option to reshape for use in Keras RNN
    if mean:
        vars = np.unique(df.variable.values)
    else:
        vars = np.unique(df.variable.values[df.variable != 'mood'])
    ids = np.unique(df.id.values)
    n_ids = ids.shape[0]
    id_df_list = [df[df.id == ids[i]] for i in range(n_ids)]

    i = 0
    c_df  = id_df_list[i]
    if mean:
        c_time_arr = create_time_arr(c_df)
        X = create_time_series(c_df, vars, time_arr=c_time_arr)
    else:
        X = create_time_series(c_df, vars)
    if mean:
        y_index = np.where(np.unique(c_df.variable.values) == 'mood')[0][0]
        y = X[y_index, :]
        X = np.delete(X, y_index, 0)
    else:
        y = c_df.value.values[c_df.variable == 'mood']
    X, y = shift_and_add_time(c_df, X, y, skip_time=mean)
    if reshape:
        X, y = rnn_reshape(X, y, 8)
        X, y = rnn_reshape_2(X, y, 8, 2)


    for i in range(1, n_ids):
        c_df = id_df_list[i]
        if mean:
            c_time_arr = create_time_arr(c_df)
            tX = create_time_series(c_df, vars, time_arr=c_time_arr)
        else:
            tX = create_time_series(c_df, vars)
        if mean:
            ty_index = np.where(np.unique(c_df.variable.values) == 'mood')[0][0]
            ty = tX[ty_index, :]
            tX = np.delete(tX, ty_index, 0)
        else:
            ty = c_df.value.values[c_df.variable == 'mood']
        tX, ty = shift_and_add_time(c_df, tX, ty, skip_time=mean)
        if reshape:
            tX, ty = rnn_reshape(tX, ty, 8)
            tX, ty = rnn_reshape_2(tX, ty, 8, 2)
        X = np.concatenate((X, tX))
        y = np.concatenate((y, ty))
        print(X.shape)
        print(y.shape)
    return X, y


def save_processed_to_csv(X, y, df):
    #Saving data to CSV file
    measurement_types = np.unique(df.variable[df.variable != 'mood'].values).tolist()
    cols = np.concatenate((measurement_types, np.array(['time', 'mood'])))
    data = np.vstack((X, y)).T
    proc_df = pd.DataFrame(data=data, columns=cols)
    return proc_df.to_csv('full_processed_data.csv', index=False)


if __name__ == "__main__":
    org_df = pd.read_csv("./Data/dataset_mood_smartphone.csv")
    org_df = org_df.dropna(axis=0, how='any')

    val_df = org_df[org_df.variable == 'circumplex.valence']
    ar_df = org_df[org_df.variable == 'circumplex.arousal']

    org_df.loc[val_df.index.values, :].replace([-2, -1, 0, 1, 2], [1, 2, 3, 4, 5])
    org_df.loc[ar_df.index.values, :].replace([-2, -1, 0, 1, 2], [1, 2, 3, 4, 5])

    org_df[org_df.variable == 'circumplex.valence'] = org_df.loc[val_df.index.values, :].replace([-2, -1, 0, 1, 2],
                                                                                                 [1, 2, 3, 4, 5])
    org_df[org_df.variable == 'circumplex.arousal'] = org_df.loc[ar_df.index.values, :].replace([-2, -1, 0, 1, 2],
                                                                                                [1, 2, 3, 4, 5])

    df = org_df.copy()

    ids = np.unique(df.id.values)
    user_df = df[df.id == ids[0]]

    X, y = merge_user_data(df, True, mean=True)
    np.save('X', X)
    np.save('y', y)
    #save_processed_to_csv(X, y, df)
