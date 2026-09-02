import pandas as pd 
import numpy as np
 
df=pd.read_csv("D:/ML Course/Datasets/House Price Prediction Dataset.csv")
print(df.head())

print(df.isnull().sum())
print(df.isnull().sum().sum())

input=df[['Area','Bedrooms']]
output=df[['Price']]
print(input)
print(output)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(input,output,test_size=0.2,random_state=10)
print(x_train)
print(y_train)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.fit_transform(x_test)
print(x_train)
print(x_test)

#train the model using k-nearest neughbour regressor 
from sklearn.neighbors import KNeighborsRegressor
model=KNeighborsRegressor(n_neighbors=3)
model.fit(x_train,y_train)

#test the model and find the accuracy of the model
y_pred = model.predict(x_test)
print("Support Vector Regression score:",model.score(x_test,y_test))

from sklearn.metrics import r2_score
print("Support Vector Regression R2 Score:", r2_score(y_test, y_pred)) 

from sklearn.metrics import mean_squared_error
print("MSE:", mean_squared_error(y_test, y_pred))
