import pandas as pd
import gc
import numpy as np

#for log submission

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

