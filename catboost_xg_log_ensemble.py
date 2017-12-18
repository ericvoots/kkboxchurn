import catboost
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import gc
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import pandas as pd

import numpy as np

#add
#stopping dev due to xgboost issues
df_train_data = pd.read_csv('data\\full_train.csv',nrows=10000)
print(len(df_train_data))

df_train = pd.read_csv('data\\train.csv')
df_train2 = pd.read_csv('data\\train_v2.csv')
df_train = df_train.append(df_train2)
del df_train2
gc.collect()
target = df_train['is_churn']
target = target[0:9999]

print('\nTraining Data columns\n', df_train_data.columns)

df_train_data = df_train_data.drop(['Unnamed: 0','date'], axis=1)


df_train_data = df_train_data.fillna(0)

print(df_train_data.head(5))

test = pd.read_csv('data\\full_test.csv')
print(len(test))

for col in df_train_data.select_dtypes(include=['object']).columns:
    df_train_data[col] = df_train_data[col].astype('category')
    test[col] = test[col].astype('category')

# Encoding categorical features
for col in df_train_data.select_dtypes(include=['category']).columns:
    df_train_data[col] = df_train_data[col].cat.codes
    test[col] = test[col].cat.codes


xgb = XGBClassifier(max_depth=4, n_estimators=300, learning_rate=0.05)
xgb.fit(df_train, target)
print('xggboost score', xgb.score(df_train_data, target))

#predictions = gbm.predict(test_X)

