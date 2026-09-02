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

#train the model using ridge regression
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV
ridge_regressor=Ridge()

#find best value of alpha which is originally known as lambda
parameters={'alpha':[1,2,5,10,20,30,40,50,60,70,80,90]}
ridgecv=GridSearchCV(ridge_regressor,parameters,scoring='neg_mean_squared_error',cv=5)
ridgecv.fit(x_train, y_train)                 #model is train perfectly

print(ridgecv.best_params_)
print('score:',ridgecv.best_score_)
ridge_pred=ridgecv.predict(x_test)

#seen variance in grapgh
import seaborn as sns
sns.displot(ridge_pred - y_test.values.ravel(), kind='kde')
plt.show()

from sklearn.metrics import r2_score
print("Score:", ridgecv.score(x_test, y_test))
print('r2 score:',r2_score(y_test,ridge_pred))

# For Ridge
train_r2 = ridgecv.score(x_train, y_train)
test_r2  = ridgecv.score(x_test, y_test)
print("Train R2:", train_r2)
print("Test R2:", test_r2)