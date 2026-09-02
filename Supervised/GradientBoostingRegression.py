import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#That have create dataset that has x has input and y has the column 
X=np.random.rand(100,1)-0.5
Y=3*X[:,0]**2+0.05*np.random.randn(100)

df=pd.DataFrame()
df['X']=X.reshape(100)
df['Y']=Y
print(df)

#plot the column X and Y in graph
plt.scatter(df['X'],df['Y'])
plt.title('X vs Y')
plt.show()

#step 1:predictions as the mean of the target value
df['pred1']=df['Y'].mean()
print(df)

#Residual error: difference between predicted value and actual error 
df['res1']=df['Y']-df['pred1']
print(df)

#see the prediction line of the base model
plt.scatter(df['X'],df['Y'])
plt.plot(df['X'],df['pred1'],color='red')
plt.show()

#create the decision tree 1 consider inputs X and output as residual error(res1)
from sklearn.tree import DecisionTreeRegressor
tree1=DecisionTreeRegressor(max_leaf_nodes=8)
tree1.fit(df['X'].values.reshape(100,1),df['res1'].values) 

#show the Decision Tree 1
from sklearn.tree import plot_tree
plot_tree(tree1)
plt.show()

#generating the X_test list
X_test=np.linspace(-0.5,0.5,500)

#predicted output
y_pred=0.218158+tree1.predict(X_test.reshape(500,1))

#see the new prediction line of the model 
plt.figure(figsize=(14,4))
plt.subplot(121)
plt.plot(X_test,y_pred,linewidth=2,color='red')
plt.scatter(df['X'],df['Y'])
plt.show()

#add the pred2 column in dataset
df['pred2']=0.218158+tree1.predict(df['X'].values.reshape(100,1))
print(df)
df['res2']=df['Y']=df['pred2']

#build the tree 2 in this input as X and output as resisual error2(res2)
tree2=DecisionTreeRegressor(max_leaf_nodes=8)
tree2.fit(df['X'].values.reshape(100,1),df['res2'].values)

y_pred=0.218158+sum(regressor.predict(X_test.reshape(-1,1))for regressor in [tree1,tree2])

#seen the new prediction line of thr model 
plt.figure(figsize=(14,4))
plt.subplot(121)
plt.plot(X_test,y_pred,linewidth=2,color='red')
plt.scatter(df['X'],df['Y'])
plt.title('X and Y')
plt.show()