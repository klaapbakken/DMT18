import numpy as np
import pandas as pd

from utility import t_delta, rnn_reshape, rnn_reshape_2, create_time_arr, manipulate_df_vals, extract_next_day_average


def create_time_series(user_df, u_vars, rm_mood=False, shift=False, add_id=False, add_date=False,
                       time_arr=None, nan=False, mask=False, day_avg=False):
    """Creates a matrix containing information on activity between specific time points

    user_df - the data frame corresponding to a specific user, as seen when importing data from
    dataset_mood_smartphone.csv
    u_vars - numpy array containing information on variables that should be considered
    time_arr - numpy array containing time points (default = None). If None, then time points of mood measurement
    is used
    """

    # Getting time grid

    if time_arr is None:
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

    # Variables to sum, variables to average
    m_vars = np.array(['activity', 'circumplex.valence', 'circumplex.arousal', 'mood'])
    s_vars = np.array(list(filter(lambda x: x not in m_vars, u_vars)))

    # Covariate matrix
    X = np.zeros((len(u_vars), n))
    if nan or mask:
        X.fill(np.nan)
    # Indices of values that should not be included next loop
    drop_list = np.array([])

    j = 0
    # Iteration through time array
    for t in tg:
        # Indices of time points preceding current time point
        prev_t_i = np.array([k for k in range(l)
                             if t_delta(var_times[k], t) >= 0
                             and k not in drop_list])
        if len(prev_t_i) == 0:
            continue
        # Variables that correspond to earlier time points
        prev_vars = var_arr[prev_t_i]
        i = 0
        for u_var in u_vars:
            # Indices that correspond to values that should be included in design matrix this loop
            prev_m_i = np.nonzero(np.isin(prev_vars, u_var))[0]
            if len(prev_m_i) != 0:
                # Indices in array for all values
                m_i = prev_t_i[prev_m_i]
                # To sum
                if u_var in s_vars:
                    X[i, j] = np.sum(var_vals[m_i])
                # To average
                elif u_var in m_vars:
                    X[i, j] = np.mean(var_vals[m_i])
                drop_list = np.concatenate((drop_list, m_i))
            i += 1
        j += 1
    if day_avg:
        m_pos = np.where(u_vars == 'mood')[0][0]
        X, y = extract_next_day_average(X, tg, user_df, m_pos)
    else:
        y_index = np.where(u_vars == 'mood')[0][0]
        y = X[y_index, :]
    if add_id:
        X = np.vstack((X, np.repeat(user_df.id.values[0], X.shape[1])))
    if add_date:
        X = np.vstack((X, tg))
    if shift:
        X, y = shift_and_add_time(X, y, tg, skip_time=(time_arr is None))
    if rm_mood:
        X = np.delete(X, y_index, 0)
    if mask:
        mask_matrix = (X == np.nan).astype('int')
        X = np.vstack((X, mask_matrix))
    if mask and not nan:
        X = np.nan_to_num(X)
    return X, y


def shift_and_add_time(X, y, time_arr, l=1, skip_time=True):
    # Adding time since last mood measurement as feature
    # Shifting response one position to the right
    sX = X[:, :-l]
    if not skip_time:
        t_d = np.array([t_delta(time_arr[i], time_arr[i + 1]) for i in range(y.shape[0] - l)])
        sX = np.vstack((sX, t_d))
    sy = y[l:]

    return sX, sy


def merge_user_data(df, reshape, rm_mood=True, add_id=False, add_date=False, shift=False,
                    l=8, seq_shift=1, collapse=False, m_tg=False, nan=False, mask=False, day_avg=False):
    # Collecting data from all users in a data frame into a feature matrix
    # Option to reshape for use in Keras RNN
    vars = np.unique(df.variable.values)
    ids = np.unique(df.id.values)
    n_ids = ids.shape[0]
    id_df_list = [df[df.id == ids[i]] for i in range(n_ids)]

    i = 0
    c_df = id_df_list[i]
    if collapse:
        c_time_arr = create_time_arr(c_df, m_tg=m_tg)
        X, y = create_time_series(c_df, vars, rm_mood=rm_mood, add_id=add_id, add_date=add_date,
                                  shift=shift, time_arr=c_time_arr, nan=nan, mask=mask, day_avg=day_avg)
    else:
        X, y = create_time_series(c_df, vars, rm_mood=rm_mood, add_id=add_id, add_date=add_date,
                                  shift=shift, nan=nan, mask=mask, day_avg=day_avg)
    if reshape:
        X, y = rnn_reshape(X, y, l)
        X, y = rnn_reshape_2(X, y, l, seq_shift)
    for i in range(1, n_ids):
        c_df = id_df_list[i]
        print(i)
        if collapse:
            c_time_arr = create_time_arr(c_df, m_tg=m_tg)
            tX, ty = create_time_series(c_df, vars, rm_mood=rm_mood, add_id=add_id, add_date=add_date,
                                        shift=shift, time_arr=c_time_arr, nan=nan, mask=mask, day_avg=day_avg)
        else:
            tX, ty = create_time_series(c_df, vars, rm_mood=rm_mood, add_id=add_id, add_date=add_date,
                                        shift=shift, nan=nan, mask=mask, day_avg=day_avg)
        if reshape:
            tX, ty = rnn_reshape(tX, ty, l)
            tX, ty = rnn_reshape_2(tX, ty, l, seq_shift)
        if reshape:
            X = np.concatenate((X, tX))
            y = np.concatenate((y, ty))
        else:
            X = np.hstack((X, tX))
            y = np.hstack((y, ty))
        print(X.shape)
        print(y.shape)
    if reshape:
        y = y[::seq_shift]
    print('Final arrays: ')
    print(X.shape)
    print(y.shape)
    return X, y, vars


def save_processed_to_csv(fname, X, u_vars, rm_mood=False, add_date=False, add_id=False, add_t_delta=False):
    # Saving data to CSV file
    if rm_mood:
        cols = np.array([u_vars[i] for i in range(len(u_vars)) if u_vars[i] != 'mood'])
    else:
        cols = u_vars
    if add_id:
        cols = np.hstack((cols, ['id']))
    if add_date:
        cols = np.hstack((cols, ['date']))
    if add_t_delta:
        cols = np.hstack((cols, ['tdelta']))
    data = X.T
    proc_df = pd.DataFrame(data=data, columns=cols)
    return proc_df.to_csv(fname, index=False)
