import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("D:\ML Course\Datasets\Student_Performance.csv")
print(df.head(3))
print(df.shape)
print(df.describe())
print(df.isnull().sum())
print(df.columns)

input=df[['Hours Studied', 'Previous Scores','Sleep Hours', 'Sample Question Papers Practiced']]
target=df[['Performance Index']]
print(input.head())
print(target.head())

#split dataset as train data and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(input,target,test_size=0.1,random_state=10)
print(x_train)
print(x_test)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
Lasso=Lasso()

#find the best value of alpha
Parameters={'alpha':[1,2,3,4,5,10,20,30,40,50,60,15,6,11]}
lassocv=GridSearchCV(Lasso,Parameters,scoring='neg_mean_squared_error',cv=5)
lassocv.fit(x_train,y_train)

print(lassocv.best_params_)
print(lassocv.best_score_)

lasso_pred=lassocv.predict(x_test)
from sklearn.metrics import r2_score
print('r2 score:',r2_score(y_test,lasso_pred))

# For Ridge
train_r2 = lassocv.score(x_train, y_train)
test_r2  = lassocv.score(x_test, y_test)
print("Train R2:", train_r2)
print("Test R2:", test_r2)
