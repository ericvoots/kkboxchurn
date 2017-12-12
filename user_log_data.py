import pandas as pd
import gc
#create user log dataframe
df_user = pd.read_csv('data\\user_logs.csv',nrows=10000)
df_user2 = pd.read_csv('data\\user_logs_v2.csv',nrows=10000)
df_user_all = df_user.append(df_user2)
print('user length\n', len(df_user))
print('\nuser length 2\n', len(df_user2))
del df_user, df_user2
gc.collect()
print('\nuserlength all\n', len(df_user_all))

#lagged changes
df_user_all['num_25_l1'] = df_user_all.groupby('msno')['num_25'].shift(1)
df_user_all['num_50_l1'] = df_user_all.groupby('msno')['num_50'].shift(1)
df_user_all['num_75_l1'] = df_user_all.groupby('msno')['num_75'].shift(1)
df_user_all['num_985_l1'] = df_user_all.groupby('msno')['num_985'].shift(1)
df_user_all['num_100_l1'] = df_user_all.groupby('msno')['num_100'].shift(1)

#percentage changes
df_user_all['num_25_pct'] = (df_user_all['num_25'] - df_user_all['num_25_l1']) / (df_user_all['num_25_l1'])
df_user_all['num_50_pct'] = (df_user_all['num_50'] - df_user_all['num_50_l1']) / (df_user_all['num_50_l1'])
df_user_all['num_75_pct'] = (df_user_all['num_75'] - df_user_all['num_75_l1']) / (df_user_all['num_75_l1'])
df_user_all['num_985_pct'] = (df_user_all['num_985'] - df_user_all['num_985_l1']) / (df_user_all['num_985_l1'])
df_user_all['num_100_pct'] = (df_user_all['num_100'] - df_user_all['num_100_l1']) / (df_user_all['num_100_l1'])
