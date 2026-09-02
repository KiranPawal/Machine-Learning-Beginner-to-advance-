import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv("D:\ML Course\Datasets\Car_price_prediction.csv")
print(df.head(4))
print(df.columns)

#Remove all missing value raw from dataset
print(df.isnull().sum())
df.dropna(inplace=True)

#data cleaning
X=df[['Brand', 'Year', 'Fuel Type','Mileage', 'Condition']]
Y=df[['Price']]
print(X.head())
print(Y.head())

X.rename(columns={'Fuel type': 'Fuel'}, inplace=True)

#Enocode the gender and position column using LabelEncoder
print('\n')
en_data=[['Brand','Fuel Type','Condition']]
from sklearn.preprocessing import LabelEncoder
la_brand = LabelEncoder()
la_fuel = LabelEncoder()
la_Condition = LabelEncoder()
X['Brand'] = la_brand.fit_transform(X['Brand'])
X['Fuel Type'] = la_fuel.fit_transform(X['Fuel Type'])
X['Condition'] = la_Condition.fit_transform(X['Condition'])

print(X.head(5))

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=20)         #random_state controls how the data is randomly split.
print(("x_train:",len(x_train)))
print(("x_test:",len(x_test)))

#train the model 
from sklearn.tree import DecisionTreeRegressor
model=DecisionTreeRegressor(criterion='squared_error')
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
print("Support Vector Regression score:",model.score(x_test,y_test))

from sklearn.metrics import r2_score
print("Support Vector Regression R2 Score:", r2_score(y_test, y_pred)) 

from sklearn.metrics import mean_squared_error
print("MSE:", mean_squared_error(y_test, y_pred))

print(y_test)
print('\n')
print(y_pred)