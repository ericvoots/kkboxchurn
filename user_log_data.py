import pandas as pd
import gc
import numpy as np
#create user log dataframe
df_user = pd.read_csv('data\\user_logs.csv')

#split user logs up then combine, dont combine due to RAM limitations

print('user length\n', len(df_user))

#lagged changes for first log file
#percentage changes
df_user['num_25_l1'] = df_user.groupby('msno')['num_25'].shift(1)
df_user['num_25_pct'] = (df_user['num_25'] - df_user['num_25_l1']) / (df_user['num_25_l1'])
df_user = df_user.drop(['num_25_l1'], axis=1)
gc.collect()
df_user['num_50_l1'] = df_user.groupby('msno')['num_50'].shift(1)
df_user['num_50_pct'] = (df_user['num_50'] - df_user['num_50_l1']) / (df_user['num_50_l1'])
df_user = df_user.drop(['num_50_l1'], axis=1)
gc.collect()
df_user['num_75_l1'] = df_user.groupby('msno')['num_75'].shift(1)
df_user['num_75_pct'] = (df_user['num_75'] - df_user['num_75_l1']) / (df_user['num_75_l1'])
df_user = df_user.drop(['num_75_l1'], axis=1)
gc.collect()
df_user['num_985_l1'] = df_user.groupby('msno')['num_985'].shift(1)
df_user['num_985_pct'] = (df_user['num_985'] - df_user['num_985_l1']) / (df_user['num_985_l1'])
df_user = df_user.drop(['num_985_l1'], axis=1)
gc.collect()
df_user['num_100_l1'] = df_user.groupby('msno')['num_100'].shift(1)
df_user['num_100_pct'] = (df_user['num_100'] - df_user['num_100_l1']) / (df_user['num_100_l1'])
df_user = df_user.drop(['num_100_l1'], axis=1)
gc.collect()
df_user['total_secs_l1'] = df_user.groupby('msno')['total_secs'].shift(1)
df_user['total_secs_pct'] = (df_user['total_secs'] - df_user['total_secs_l1']) / (df_user['total_secs_l1'])
df_user = df_user.drop(['total_secs_l1'], axis=1)
gc.collect()
df_user = df_user.fillna(0)
print('finished lagged variables')
print('finished pct variables')
df_user = df_user.fillna(0)
df_user = df_user.replace(np.inf, 0)
df_user = df_user.replace(-np.inf, 0)


#indicator of last user log
df_user = df_user.sort_values(['msno','date'], ascending=[True, False])
df_user['last_user'] = (df_user.msno != df_user.msno.shift(-1)).astype(int)
df_user = df_user[df_user.last_user == 1]
gc.collect()

print(df_user.head(5))

#second log
#percentage changes
df_user2 = pd.read_csv('data\\user_logs_v2.csv')

df_user2['num_25_l1'] = df_user2.groupby('msno')['num_25'].shift(1)
df_user2['num_25_pct'] = (df_user2['num_25'] - df_user2['num_25_l1']) / (df_user2['num_25_l1'])
df_user2 = df_user2.drop(['num_25_l1'], axis=1)
gc.collect()
df_user2['num_50_l1'] = df_user2.groupby('msno')['num_50'].shift(1)
df_user2['num_50_pct'] = (df_user2['num_50'] - df_user2['num_50_l1']) / (df_user2['num_50_l1'])
df_user2 = df_user2.drop(['num_50_l1'], axis=1)
gc.collect()
df_user2['num_75_l1'] = df_user2.groupby('msno')['num_75'].shift(1)
df_user2['num_75_pct'] = (df_user2['num_75'] - df_user2['num_75_l1']) / (df_user2['num_75_l1'])
df_user2 = df_user2.drop(['num_75_l1'], axis=1)
gc.collect()
df_user2['num_985_l1'] = df_user2.groupby('msno')['num_985'].shift(1)
df_user2['num_985_pct'] = (df_user2['num_985'] - df_user2['num_985_l1']) / (df_user2['num_985_l1'])
df_user2 = df_user2.drop(['num_985_l1'], axis=1)
gc.collect()
df_user2['num_100_l1'] = df_user2.groupby('msno')['num_100'].shift(1)
df_user2['num_100_pct'] = (df_user2['num_100'] - df_user2['num_100_l1']) / (df_user2['num_100_l1'])
df_user2 = df_user2.drop(['num_100_l1'], axis=1)
gc.collect()
df_user2['total_secs_l1'] = df_user2.groupby('msno')['total_secs'].shift(1)

df_user2['total_secs_pct'] = (df_user2['total_secs'] - df_user2['total_secs_l1']) / (df_user2['total_secs_l1'])
df_user2 = df_user2.drop(['total_secs_l1'], axis=1)
gc.collect()
df_user2 = df_user2.fillna(0)
print('finished lagged variables 2nd')
print('finished pct variables 2nd')

df_user2 = df_user2.fillna(0)
df_user2 = df_user2.replace(np.inf, 0)
df_user2 = df_user2.replace(-np.inf, 0)

#indicator of last user log
df_user2 = df_user2.sort_values(['msno','date'], ascending=[True, False])
df_user2['last_user'] = (df_user2.msno != df_user2.msno.shift(-1)).astype(int)
df_user2 = df_user2[df_user2.last_user == 1]

#append second user log and then check for dups
df_user_all = df_user.append(df_user2)
del df_user, df_user2
gc.collect()
df_user_all = df_user_all.drop(['last_user'], axis=1)
gc.collect()
df_user_all.sort_values(['msno','date'], ascending=[True, False])
df_user_all['last_user_all'] = (df_user_all.msno != df_user_all.msno.shift(-1)).astype(int)
df_user_all = df_user_all[df_user_all.last_user_all == 1]
gc.collect()

print(df_user_all.head(5))
df_user_all.to_csv('data\\user_logs_all.csv')
print('Done')
