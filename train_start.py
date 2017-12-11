import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import ShuffleSplit
import gc

#original start from kernel:
#https://www.kaggle.com/talysacc/lgbm-starter-lb-0-23434

df_train = pd.read_csv('data\\train.csv',dtype={'is_churn' : bool,'msno' : str})
df_members = pd.read_csv('data\\members_v3.csv',dtype={'registered_via' : np.uint8})

df_train = pd.merge(left = df_train,right = df_members,how = 'left',on=['msno'])

del df_members
df_train.head()

df_transactions = pd.read_csv('data\\transactions.csv', dtype={   'payment_plan_days': np.uint8,
                                                                  'plan_list_price': np.uint8,
                                                                  'actual_amount_paid': np.uint8,
                                                                  'is_auto_renew': np.bool,
                                                                  'is_cancel': np.bool})

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

# In the description the bd column is said to be in a very wide range
# So I decided to clip it just to store it as a smaller type
df_train['bd'].clip(0, 100)
df_train['bd'].fillna(0, inplace=True)
df_train['bd'].astype(np.uint8, inplace=True)

df_train.head()

bst = None

df_train['gender'] = df_train['gender'].astype('category')

print(df_train.head(5))
'''
for train_indices, val_indices in ShuffleSplit(n_splits=1, test_size=0.1, train_size=0.4).split(df_train):
    train_data = lgb.Dataset(df_train.drop(['msno', 'is_churn'], axis=1).loc[train_indices, :],
                             label=df_train.loc[train_indices, 'is_churn'])
    val_data = lgb.Dataset(df_train.drop(['msno', 'is_churn'], axis=1).loc[val_indices, :],
                           label=df_train.loc[val_indices, 'is_churn'])

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting': 'gbdt',
        'learning_rate': 0.1,  # small learn rate, large number of iterations
        'verbose': 0,
        'num_leaves': 108,
        'bagging_fraction': 0.95,
        'bagging_freq': 1,
        'bagging_seed': 1,
        'feature_fraction': 0.9,
        'feature_fraction_seed': 1,
        'max_bin': 128,
        'max_depth': 10,
        'num_rounds': 200,
    }

    bst = lgb.train(params, train_data, 100, valid_sets=[val_data])

    df_test = pd.read_csv('data\\sample_submission_zero.csv', dtype={'msno': str})
    df_members = pd.read_csv('data\\members_v3.csv', dtype={'registered_via': np.uint8})
    df_test = pd.merge(left=df_test, right=df_members, how='left', on=['msno'])

    del df_members

    df_transactions = pd.read_csv('data\\transactions.csv', dtype={   'payment_plan_days': np.uint8,
                                                                      'plan_list_price': np.uint8,
                                                                      'actual_amount_paid': np.uint8,
                                                                      'is_auto_renew': np.bool,
                                                                      'is_cancel': np.bool})

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
    del df_transactions

    df_test['bd'].clip(0, 100)
    df_test['bd'].fillna(0, inplace=True)
    df_test['bd'] = df_test['bd'].astype(np.uint8)

    predictions = bst.predict(df_test.drop(['msno', 'is_churn'], axis=1))
    df_test['is_churn'] = predictions
    df_test.drop(['city', 'bd', 'gender', 'registered_via', 'registration_init_time', 'expiration_date', 'total_order',
                  'plan_net_worth', 'mean_payment_each_transaction', 'total_actual_payment', 'cancel_times'], axis=1,
                 inplace=True)
    df_test.to_csv('submissions.csv', index=False)
'''
print("DONE")
