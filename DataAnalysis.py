import pandas as pd
import numpy as np

train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")

train.head()
test.head()

#データサイズを確認
print('train shape:',train.shape, '\n')
print('test shape:',test.shape, '\n')

def lackdata(df):
    null_val = df.isnull().sum()
    percent = 100*df.isnull().sum()/len(df)
    lackdata = pd.concat([null_val, percent], axis = 1)
    lackdata_table = lackdata.rename(columns = {0:'lack', 1:'%'})
    return lackdata_table

#欠損数を確認
print('lack data of train.csv\n')
print(lackdata(train),'\n')
print('lack data of test.csv\n')
print(lackdata(test))

#各コラムのワード出現数をファイル出力
keyword_train = train['keyword'].value_counts(sort = False)
keyword_train.to_csv('keyword_train.csv')
keyword_test = test['keyword'].value_counts(sort = False)
keyword_test.to_csv('keyword_test.csv')

location_train = train['location'].value_counts(sort = False)
location_train.to_csv('location_train.csv')
location_test = test['location'].value_counts(sort = False)
location_test.to_csv('location_test.csv')



