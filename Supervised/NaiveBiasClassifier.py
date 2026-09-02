import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("D:\ML Course\Datasets\Tiatanic.csv")
print(df.head(3))

print(df.isnull().sum())
print(df.isnull().sum().sum())
print(df.shape)

#Fare column represents the amount of money in British pounds (£) that a passenger paid for their ticket
print(df.columns)
inputs=df[['Pclass', 'Sex', 'Age','Fare']]   #create seperate input dataset
print(inputs.head(3))
target=df[['Survived']]              #create seperate target dataset
print(target.head(3))

#processing on input data and remmove outliers
print(inputs.describe())
sns.boxplot(x="Fare",data=inputs) 
plt.show()

sns.displot(inputs["Fare"])
min_range=inputs["Fare"].mean()-(3*inputs["Fare"].std())
max_range=inputs["Fare"].mean()+(3*inputs["Fare"].std())
print(min_range,max_range)
input=inputs[inputs["Fare"]<=max_range]
sns.boxplot(x="Fare",data=input) 
plt.show()

print(input.head())
print(input.describe())

#perform encoding on sex data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
input["Sex"] = le.fit_transform(input["Sex"])

print(input.head())
print(target.head())

# If you have cleaned input (e.g., dropped missing values)
target = target.loc[input.index]

#Train the model using Naive Bias classifier
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(input,target,test_size=0.2,random_state=42)
print(("x_train:",len(x_train)))
print(("x_test:",len(x_test)))
print('\n')

#model train using Gaussian naive bias algorithm
from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(x_train,y_train)

#model train using MultinomialNB naive bias algorithm
from sklearn.naive_bayes import MultinomialNB
model2=MultinomialNB()
model2.fit(x_train,y_train)

#model train using BernoulliNB naive bias algorithm
from sklearn.naive_bayes import BernoulliNB
model3=BernoulliNB()
model3.fit(x_train,y_train)


#test the model
sample = np.array([[3, 0, 22, 7.25]])  # 2D: 1 row, 4 columns
print(model.predict(sample))
print('\n')

#score of the models
print("Gaussian model Score:",model.score(x_test,y_test))
print("MultinomialNB model Score:",model2.score(x_test,y_test))
print("BernoulliNB model Score:",model3.score(x_test,y_test))
print('\n')

#precision score of models
from sklearn.metrics import precision_score
y_pred = model.predict(x_test)
y_pred2 = model2.predict(x_test)
y_pred3 = model3.predict(x_test)
print("Gaussian model Precision score:",precision_score(y_test, y_pred))
print("MultinomialNB model Precision score:",precision_score(y_test, y_pred2))
print("BernoulliNB model Precision score:",precision_score(y_test, y_pred3))