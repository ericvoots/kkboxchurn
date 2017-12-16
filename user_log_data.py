import pandas as pd
import gc
import numpy as np
#create user log dataframe
df_user = pd.read_csv('data\\user_logs.csv',nrows=10000)

#split user logs up then combine, dont combine due to RAM limitations

print('user length\n', len(df_user))



#lagged changes for first log file
df_user['num_25_l1'] = df_user.groupby('msno')['num_25'].shift(1)
df_user['num_50_l1'] = df_user.groupby('msno')['num_50'].shift(1)
df_user['num_75_l1'] = df_user.groupby('msno')['num_75'].shift(1)
df_user['num_985_l1'] = df_user.groupby('msno')['num_985'].shift(1)
df_user['num_100_l1'] = df_user.groupby('msno')['num_100'].shift(1)
df_user['total_secs_l1'] = df_user.groupby('msno')['total_secs'].shift(1)
df_user = df_user.fillna(0)
print('finished lagged variables')

#percentage changes
df_user['num_25_pct'] = (df_user['num_25'] - df_user['num_25_l1']) / (df_user['num_25_l1'])
df_user = df_user.drop(['num_25_l1'], axis=1)
gc.collect()
df_user['num_50_pct'] = (df_user['num_50'] - df_user['num_50_l1']) / (df_user['num_50_l1'])
df_user = df_user.drop(['num_50_l1'], axis=1)
gc.collect()
df_user['num_75_pct'] = (df_user['num_75'] - df_user['num_75_l1']) / (df_user['num_75_l1'])
df_user = df_user.drop(['num_75_l1'], axis=1)
gc.collect()
df_user['num_985_pct'] = (df_user['num_985'] - df_user['num_985_l1']) / (df_user['num_985_l1'])
df_user = df_user.drop(['num_985_l1'], axis=1)
gc.collect()
df_user['num_100_pct'] = (df_user['num_100'] - df_user['num_100_l1']) / (df_user['num_100_l1'])
df_user = df_user.drop(['num_100_l1'], axis=1)
gc.collect()
df_user['total_secs_pct'] = (df_user['total_secs'] - df_user['total_secs_l1']) / (df_user['total_secs_l1'])
df_user = df_user.drop(['total_secs_l1'], axis=1)
gc.collect()
df_user = df_user.fillna(0)
df_user = df_user.replace(np.inf, 0)
df_user = df_user.replace(-np.inf, 0)
print('finished pct variables')


#indicator of last user log
df_user.sort_values(['msno','date'], ascending=[True, False])
df_user['last_user'] = (df_user.msno != df_user.msno.shift(-1)).astype(int)
df_user = df_user[df_user.last_user == 1]
gc.collect()

print(df_user.head(5))

#second log
df_user2 = pd.read_csv('data\\user_logs_v2.csv',nrows=10000)


print('Done')
