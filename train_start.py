import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import ShuffleSplit
import gc
from sklearn.linear_model import LogisticRegression
import catboost


#original start from kernel:
#https://www.kaggle.com/talysacc/lgbm-starter-lb-0-23434

def gender_to_numeric(x):
    if x == 'male':
        return 1
    elif x == 'female':
        return 2
    else:
        return 0

df_train = pd.read_csv('data\\train.csv')
df_train2 = pd.read_csv('data\\train_v2.csv')
df_train = df_train.append(df_train2)
del df_train2
gc.collect()
df_members = pd.read_csv('data\\members_v3.csv',dtype={'registered_via' : np.uint8})

df_train = pd.merge(left = df_train,right = df_members,how = 'left',on=['msno'])

del df_members
gc.collect()
df_train.head()

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

df_transactions = pd.merge(left=df_train, right=df_transactions, how='left', on='msno')
grouped = df_transactions.copy().groupby('msno')

df_stats = grouped.agg({'msno': {'total_order': 'count'},
                        'plan_list_price': {'plan_net_worth': 'sum'},
                        'actual_amount_paid': {'mean_payment_each_transaction': 'mean',
                                               'total_actual_payment': 'sum'},
                        'is_cancel': {'cancel_times': lambda x: sum(x == 1)}})

df_stats.columns = df_stats.columns.droplevel(0)
df_stats.reset_index(inplace=True)
df_train = pd.merge(left=df_train, right=df_stats, how='left', on='msno')

del df_transactions, df_stats
gc.collect()

# In the description the bd column is said to be in a very wide range
# So I decided to clip it just to store it as a smaller type
df_train['bd'].clip(0, 100)
df_train['bd'].fillna(0, inplace=True)
df_train['bd'].astype(np.uint8, inplace=True)

df_train.head()

bst = None

df_train['gender'] = df_train['gender'].apply(gender_to_numeric)

df_user_logs = pd.read_csv('data\\user_logs_all.csv')

df_train = pd.merge(left=df_train, right=df_user_logs, how='left', on='msno')

del df_user_logs
gc.collect()

print(df_train.head(5))

clf_log = LogisticRegression(C=1, penalty='l2')

target = df_train['is_churn']

df_train = df_train.drop(['msno','is_churn','last_user_all'],axis=1)

df_train = df_train.fillna(0)
df_train = df_train.replace(np.inf, 0)
df_train = df_train.replace(-np.inf, 0)
df_train.to_csv('data\\full_train.csv')
#not needed
#target.to_csv('data\\full_target.csv')
clf_log.fit(df_train, target)

print('\nScore of Log L2 and C=1\n', clf_log.score(df_train, target))

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
df_test.to_csv('data\\full_test.csv')
predictions = clf_log.predict_proba(df_test)

predictions = pd.DataFrame(predictions)

predictions.to_csv('data\\predictions_log.csv')


print("DONE")
