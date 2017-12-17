import pandas as pd
import gc
import numpy as np

def gender_to_numeric(x):
    if x == 'male':
        return 1
    elif x == 'female':
        return 2
    else:
        return 0


df_pred = pd.read_csv('data\\predictions_log.csv')

print(df_pred.head(5))

df_sample = pd.read_csv('data\\sample_submission_v2.csv')

df_sample = df_sample.drop(['is_churn'],axis=1)

df_submit = pd.merge(df_pred, df_sample, left_index=True, right_index=True)
print('test spot \n',df_submit.head(5))
df_submit = df_submit.drop(['Unnamed: 0'],axis=1)
df_submit = df_submit.rename(index=str, columns={"1": "is_churn","msno": "msno"})
df_submit = df_submit[['msno','is_churn']]
print('last spot\n',df_submit.head(5))

df_submit.to_csv('data\\log_predictions.csv', index= False)

#take a look at test df

df_test = pd.read_csv('data\\sample_submission_v2.csv')
df_members = pd.read_csv('data\\members_v3.csv')
df_test = pd.merge(left=df_test, right=df_members, how='left', on=['msno'])
del df_members
gc.collect()

df_transactions = pd.read_csv('data\\transactions.csv', dtype={'payment_plan_days': np.uint8,
                                                                  'plan_list_price': np.uint8,
                                                                  'actual_amount_paid': np.uint8,
                                                                  'is_auto_renew': np.bool,
                                                                  'is_cancel': np.bool})
df_transactions2 = pd.read_csv('data\\transactions_v2.csv', dtype={'payment_plan_days': np.uint8,
                                                                  'plan_list_price': np.uint8,
                                                                  'actual_amount_paid': np.uint8,
                                                                  'is_auto_renew': np.bool,
                                                                  'is_cancel': np.bool})
df_transactions = df_transactions.append(df_transactions2)
del df_transactions2
gc.collect()

df_transactions = pd.merge(left=df_test, right=df_transactions, how='left', on='msno')
grouped = df_transactions.copy().groupby('msno')

df_stats = grouped.agg({'msno': {'total_order': 'count'},
                        'plan_list_price': {'plan_net_worth': 'sum'},
                        'actual_amount_paid': {'mean_payment_each_transaction': 'mean',
                                               'total_actual_payment': 'sum'},
                        'is_cancel': {'cancel_times': lambda x: sum(x == 1)}})

df_stats.columns = df_stats.columns.droplevel(0)
df_stats.reset_index(inplace=True)

df_test = pd.merge(left=df_test, right=df_stats, how='left', on='msno')
del df_transactions, df_stats
gc.collect()

df_user_logs = pd.read_csv('data\\user_logs_all.csv')

df_test = pd.merge(left=df_test, right=df_user_logs, how='left', on='msno')

del df_user_logs
gc.collect()

df_test = df_test.drop(['msno','is_churn','last_user_all'],axis=1)

df_test['gender'] = df_test['gender'].apply(gender_to_numeric)

df_test = df_test.fillna(0)
df_test = df_test.replace(np.inf, 0)
df_test = df_test.replace(-np.inf, 0)

print(df_test.head(5))

print('DONE')