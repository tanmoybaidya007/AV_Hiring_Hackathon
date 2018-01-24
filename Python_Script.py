print("Libraries Importing... \n")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
#import tensorflow
#import seaborn as sns
import xgboost
#import catboost
#import lightgbm
#import keras
#%matplotlib inline

print("Datasets Importing... \n")

train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
print("Data Preprocessing Started... \n")
## Data Preprocessing

train.datetime=pd.to_datetime(train.datetime)
test.datetime=pd.to_datetime(test.datetime)

train["year"]=train.datetime.apply(lambda x: x.year)
train["month"]=train.datetime.apply(lambda x: x.month)
train["week"]=train.datetime.apply(lambda x: x.week)
train["day"]=train.datetime.apply(lambda x: x.day)
train["hour"]=train.datetime.apply(lambda x: x.hour)
train["quarter"]=train.datetime.apply(lambda x: x.quarter)
train["dayofyear"]=train.datetime.apply(lambda x: x.dayofyear)
train["day_diff"]=np.abs((2013-train.datetime.dt.year)*365)+np.abs((7-train.datetime.dt.month)*30)+np.abs((1-train.datetime.dt.day))

test["year"]=test.datetime.apply(lambda x: x.year)
test["month"]=test.datetime.apply(lambda x: x.month)
test["week"]=test.datetime.apply(lambda x: x.week)
test["day"]=test.datetime.apply(lambda x: x.day)
test["hour"]=test.datetime.apply(lambda x: x.hour)
test["quarter"]=test.datetime.apply(lambda x: x.quarter)
test["dayofyear"]=test.datetime.apply(lambda x: x.dayofyear)
test["day_diff"]=np.abs((2013-test.datetime.dt.year)*365)+np.abs((7-test.datetime.dt.month)*30)+np.abs((1-test.datetime.dt.day))


train=train[["ID","day_diff","year","quarter",'month','week',"dayofyear",'day','hour','temperature','var1','pressure','windspeed','var2','electricity_consumption']]
test=test[["ID","day_diff","year","quarter",'month','week','dayofyear','day','hour','temperature','var1','pressure','windspeed','var2']]

## Temperature Category 

def Temp_Category(x):
    p=[11,12,1,2,3]
    q=[4,10]
    r=[5,6,7,8,9]
    if x in p:
        return("P")
    elif x in q:
        return("Q")
    elif x in r:
        return("R")
train=pd.concat([train,pd.get_dummies(train.month.apply(lambda x: Temp_Category(x)),drop_first=True)],axis=1)
test=pd.concat([test,pd.get_dummies(test.month.apply(lambda x: Temp_Category(x)),drop_first=True)],axis=1)

## Temprature Outlier

def Temp_Outlier(x):
    if x>15.0 or x<-7.5:
        return(1)
    else:
        return(0)

train["temp_outlier"]=train.temperature.apply(lambda x: Temp_Outlier(x))
test["temp_outlier"]=test.temperature.apply(lambda x: Temp_Outlier(x))

## Windspeed Outlier

def Windspeed_Outlier(x):
    if x>10 or x<2:
        return(1)
    else:
        return(0)

train["windspeed_outlier"]=train.windspeed.apply(lambda x: Windspeed_Outlier(x))
test["windspeed_outlier"]=test.windspeed.apply(lambda x: Windspeed_Outlier(x))

## Var1_Category

def Var_Category(x):
    p=[1,2]
    q=[3,4,11,12]
    r=[5,9,10]
    s=[6,7,8]
    if x in p:
        return("D")
    elif x in q:
        return("E")
    elif x in r:
        return("F")
    elif x in s:
        return("G")

train=pd.concat([train,pd.get_dummies(train.month.apply(lambda x:Var_Category(x)),drop_first=True)],axis=1)
test=pd.concat([test,pd.get_dummies(test.month.apply(lambda x:Var_Category(x)),drop_first=True)],axis=1)

## Var1_Outlier

def Var1_Outlier(x):
    if x<-15.0 or x>12.5:
        return(1)
    else:
        return(0)

train["var1_outlier"]=train.var1.apply(lambda x: Var1_Outlier(x))
test["var1_outlier"]=test.var1.apply(lambda x: Var1_Outlier(x))

## VAR2

## Normal Label Encoding

train=pd.concat([train,pd.get_dummies(train.var2,drop_first=True)],axis=1)
test=pd.concat([test,pd.get_dummies(test.var2,drop_first=True)],axis=1)

train.drop('var2',axis=1,inplace=True)
test.drop('var2',axis=1,inplace=True)

train=train[['ID', 'day_diff', 'year', 'quarter', 'month', 'week', 'dayofyear',
       'day', 'hour','temperature','Q','R','temp_outlier','var1','E','F','G','var1_outlier','pressure','windspeed',
       'windspeed_outlier','B','C','electricity_consumption']]


test=test[['ID', 'day_diff', 'year', 'quarter', 'month', 'week', 'dayofyear',
       'day', 'hour','temperature','Q','R','temp_outlier','var1','E','F','G','var1_outlier','pressure','windspeed',
       'windspeed_outlier','B','C']]

## Special Treatment for XGBoost

def Outlier(data):
    x=data[0]
    y=data[1]
    z=data[2]
    if x+y+z>=2:
        return(1)
    else:
        return(0)

train["Outlier"]=train[['temp_outlier','var1_outlier','windspeed_outlier']].apply(lambda x:Outlier(x))
test["Outlier"]=test[['temp_outlier','var1_outlier','windspeed_outlier']].apply(lambda x:Outlier(x))


train.drop(['temp_outlier','windspeed_outlier','var1_outlier'],axis=1,inplace=True)
test.drop(['temp_outlier','windspeed_outlier','var1_outlier'],axis=1,inplace=True)

train=train[['ID', 'day_diff', 'year', 'quarter', 'month', 'week', 'dayofyear',
       'day', 'hour', 'temperature', 'Q', 'R', 'var1', 'E', 'F', 'G',
       'pressure', 'windspeed', 'B', 'C', 'Outlier','electricity_consumption',
       ]]

test=test[['ID', 'day_diff', 'year', 'quarter', 'month', 'week', 'dayofyear',
       'day', 'hour', 'temperature', 'Q', 'R', 'var1', 'E', 'F', 'G',
       'pressure', 'windspeed', 'B', 'C', 'Outlier'
       ]]

print("Modelling Phase Started... \n")

X_train,X_test,y_train,y_test=train_test_split(train.iloc[:,1:-1],train.iloc[:,-1],test_size=0.2,random_state=42)

from xgboost import XGBRegressor
model_XGB=XGBRegressor(learning_rate=0.1,n_estimators=500,max_depth=10,colsample_bytree=0.5)
model_XGB.fit(X_train,y_train)

pred_XGB=model_XGB.predict((test.iloc[:,1:]))

## Submission
submission=pd.read_csv("sample_submission_q0Q3I1Z.csv")
submission.ID=test.ID
submission.electricity_consumption=pred_XGB
submission.to_csv("Submission_XGB_500.csv",index=False)






































