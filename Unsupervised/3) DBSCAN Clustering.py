import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons

#create dataset using make_moon 
X,Y=make_moons(n_samples=250,noise=0.05)
print(X)

#plot the graph
plt.scatter(X[:,0],X[:,1])
plt.show()

#scaled the dataset
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_scaled=scaler.fit_transform(X)

#train the model for clustering
from sklearn.cluster import DBSCAN
dbscan=DBSCAN(eps=0.5)
dbscan.fit_predict(x_scaled)

print(dbscan.labels_)

#plot the graph base on train data
plt.scatter(X[:,0],X[:,1],c=dbscan.labels_)
plt.show()

