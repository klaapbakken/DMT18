from utility import manipulate_df_vals
from features import merge_user_data, save_processed_to_csv
#from rnn import train_model

import pandas as pd
import numpy as np


org_df = pd.read_csv("./Data/dataset_mood_smartphone.csv")
df = manipulate_df_vals(org_df)

users = np.unique(df.id.values)

tr_users = users[0:14]
tr_df = df.loc[df['id'].isin(tr_users)]

te_users = users[14:]
te_df = df.loc[df['id'].isin(te_users)]


col_X, col_y, u_vars = merge_user_data(df, False, l=8, seq_shift=1, mean=True)
#b_y_pred = np.roll(col_y, 1)

save_processed_to_csv(col_X, col_y, df, u_vars)

#m_X, m_y = merge_user_data(tr_df, True, l=8, seq_shift=2, mean=False)
#model = train_model(m_X, m_y, seq_shift=2)

