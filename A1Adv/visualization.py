import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from features import create_time_series, shift_and_add_time

df = pd.read_csv("./Data/dataset_mood_smartphone.csv")
df = df.dropna(axis=0, how='any')
ids = np.unique(df.id.values)

user_df = df[df.id == ids[0]]

umt = np.unique(df.variable.values[df.variable != 'mood'])
X, y = create_time_series(user_df, umt)

print(umt)

indices = [0, 13, 16, 17]
i = 0
f, axes = plt.subplots(2, 2)

powerset = [(0,0), (0,1), (1,0), (1,1)]

for index in indices:
    axes[powerset[i][0], powerset[i][1]].plot(np.arange(len(X[index, :])), X[index, :])
    axes[powerset[i][0], powerset[i][1]].set_title(umt[index])
    i += 1

plt.figure(5)
plt.plot(np.arange(len(y)), y)
plt.show()
