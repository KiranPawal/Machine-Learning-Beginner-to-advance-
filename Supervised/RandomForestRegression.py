import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("D:/ML Course/Datasets/employee_data.csv")
print(df.head())

#null values
print(df.isnull().sum().sum())

print('\n')
#apply feature scaling on the salary data
df['Salary'].fillna(df['Salary'].mean(),inplace=True)
sns.displot(df['Salary'])
plt.show()

print(df.describe())

#Enocode the gender and position column using LabelEncoder
print('\n')
en_data=[['Gender','Position']]
from sklearn.preprocessing import LabelEncoder
la_gender = LabelEncoder()
la_position = LabelEncoder()
df['Gender'] = la_gender.fit_transform(df['Gender'])
df['Position'] = la_position.fit_transform(df['Position'])

print(df['Gender'].unique())
print(df.head())
print('\n')

#spilt the training and testing data
print(df.columns)
input=df[['Gender','Experience (Years)','Position']]    #create input dataset
target=df[['Salary']]                   #create output/target dataset
print(input.head(4))
print(target.head(4))
print('\n')

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(input,target,test_size=0.2,random_state=20)
print(("x_train:",len(x_train)))
print(("x_test:",len(x_test)))
print('\n')

#train the model using the random forest regression
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(
    n_estimators=100,                 #Meaning: Number of trees in your forest.
    max_depth=15,                   #Maximum depth (number of splits) of each tree.
    random_state=20                   #Meaning: Seed for the random number generator used by the algorithm.
)

# Train model
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print("Support Vector Regression score:",model.score(x_test,y_test))

from sklearn.metrics import r2_score
print("Support Vector Regression R2 Score:", r2_score(y_test, y_pred)) 

from sklearn.metrics import mean_squared_error
print("MSE:", mean_squared_error(y_test, y_pred))