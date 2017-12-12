import pandas as pd
#create user log dataframe
df_user = pd.read_csv('data\\user_logs.csv')
df_user2 = pd.read_csv('data\\user_logs_v2.csv')
df_user_all = df_user.append(df_user2)
print('user length\n', len(df_user))
print('\nuser length 2\n', len(df_user2))
del df_user, df_user2
gc.collect()
print('\nuserlength all\n', len(df_user_all))

df_user_all['num_25_l1'] = df_user_all.groupby('')