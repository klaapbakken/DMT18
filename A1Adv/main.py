from utility import manipulate_df_vals
from features import merge_user_data, save_processed_to_csv
from rnn import rnn, fcnn
from sklearn.model_selection import KFold

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

org_df = pd.read_csv("./Data/dataset_mood_smartphone.csv")
df = manipulate_df_vals(org_df)
users = np.unique(df.id.values)
tr_users = np.random.choice(users, size=20, replace=False)
tr_df = df.loc[df['id'].isin(tr_users)]
te_users = [users[i] for i in range(len(users)) if users[i] not in tr_users]
te_df = df.loc[df['id'].isin(te_users)]

baseline = False

col_seq = True
cs_merge = False
cross_val = True

u_col_seq = False
ucs_merge = False

create_csv = False


if baseline:
    col_X, col_y, u_vars = merge_user_data(te_df, False, rm_mood=False,
                                       add_id=False, add_date=True, shift=True, l=1,
                                           seq_shift=1, collapse=True, m_tg=True)
    b_y_pred = col_X

    np.save('baseline_pred', b_y_pred)
    save_processed_to_csv('baseline.csv', col_X, u_vars, rm_mood=False, add_date=True, add_id=False, add_t_delta=True)

if col_seq:
    if cs_merge:
        tr_X_seq, tr_y, tr_u_vars = merge_user_data(tr_df, True, rm_mood=False, add_id=False, add_date=False,
                                       shift=False, l=20, seq_shift=1, collapse=True, m_tg=True, mask=True, day_avg=True)
        te_X_seq, te_y, te_u_vars = merge_user_data(te_df, True, rm_mood=False, add_id=False, add_date=False,
                                       shift=False, l=20, seq_shift=1, collapse=True, m_tg=True, mask=True, day_avg=True)
        np.save('trainX', tr_X_seq)
        np.save('trainy', tr_y)
        np.save('testX', te_X_seq)
        np.save('testy', te_y)
    else:
        tr_X_seq = np.load('trainX.npy')
        tr_y = np.load('trainy.npy')
        te_X_seq = np.load('testX.npy')
        te_y = np.load('testy.npy')
    if cross_val:
        X, y, u_vars = merge_user_data(df, True, rm_mood=False, add_id=False, add_date=False,
                                       shift=False, l=20, seq_shift=1, collapse=True, m_tg=True, mask=True, day_avg=True)
        kfold = KFold(n_splits=10, shuffle=True)
        for train, test in kfold.split(X, y):
            cvscores = []
            model = rnn(X[train], y[train])
            y_hat = np.squeeze(model.predict(X[test]))
            mae = np.mean(np.abs(te_y - y_hat))
            cvscores.append(mae)
            print(cvscores)



    te_X_seq = np.nan_to_num(te_X_seq)

    #model = rnn(tr_X_seq, tr_y)
    y_hat = np.squeeze(model.predict(te_X_seq))
    mae = np.mean(np.abs(te_y - y_hat))

    b_y = np.roll(te_y, -1)[:-1]
    b_mae = np.mean(np.abs(te_y[:-1] - b_y))

    #plt.plot(np.arange(len(b_y)), np.abs(b_y - te_y[:-1]))
    #plt.plot(np.arange(len(y_hat)), np.abs(y_hat - te_y))
    #plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Index')
    ax.set_ylabel('Mood')
    ax.set_title('Mood predictions')
    ax.text(0, 5, 'Baseline MAE: ' + str(b_mae) + '\n RNN MAE: ' + str(mae))


    ax.plot(np.arange(len(te_y)), te_y, 'r')
    ax.plot(np.arange(len(y_hat)), y_hat, 'g')
    plt.show()

    print('MAE: ', mae, '\n Baseline MAE: ', b_mae)

if u_col_seq:
    if ucs_merge:
        tr_X_seq, tr_y, tr_u_vars = merge_user_data(tr_df, True, rm_mood=False, add_id=False, add_date=False,
                                       shift=False, l=20, seq_shift=2, collapse=False, m_tg=True, mask=False, day_avg=True)
        te_X_seq, te_y, te_u_vars = merge_user_data(te_df, True, rm_mood=False, add_id=False, add_date=False,
                                       shift=False, l=20, seq_shift=2, collapse=False, m_tg=True, mask=False, day_avg=True)
        np.save('trainXu', tr_X_seq)
        np.save('trainyu', tr_y)
        np.save('testXu', te_X_seq)
        np.save('testyu', te_y)
    else:
        tr_X_seq = np.load('trainXu.npy')
        tr_y = np.load('trainyu.npy')
        te_X_seq = np.load('testXu.npy')
        te_y = np.load('testyu.npy')

    model = rnn(tr_X_seq, tr_y)
    y_hat = np.squeeze(model.predict(te_X_seq))
    mae = np.mean(np.abs(te_y - y_hat))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Index')
    ax.set_ylabel('Mood')
    ax.set_title('Mood predictions')
    ax.text(0, 5, 'RNN MAE: ' + str(mae))



    #b_y = np.roll(te_y, -1)[:-1]
    #b_mae = np.mean(np.abs(te_y[:-1] - b_y))

    #plt.plot(np.arange(len(b_y)), np.abs(b_y - te_y[:-1]))
    #plt.plot(np.arange(len(y_hat)), np.abs(y_hat - te_y))
    #plt.show()

    #plt.plot(np.arange(len(te_y)), te_y)
    #plt.plot(np.arange(len(y_hat)), y_hat)
    #plt.show()

if create_csv:
    X, y, u_vars = merge_user_data(df, False, rm_mood=False, add_id=True, add_date=True,
                                                shift=True, l=1, seq_shift=1, collapse=True, m_tg=False, nan=True, add_var=True)
    save_processed_to_csv('processed_data.csv', X, u_vars=u_vars, rm_mood=False, add_id=True, add_date=True, add_t_delta=True, add_var=True)







