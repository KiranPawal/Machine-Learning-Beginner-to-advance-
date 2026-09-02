import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

df=pd.read_csv("D:\ML Course\Datasets\Mall_Customers.csv")
print(df.head())

data=df[["Annual Income (k$)","Spending Score (1-100)"]]
print(data.head())

#create dendogram using the scipy library
import scipy.cluster.hierarchy as sc
plt.figure(figsize=(10,7))
plt.title("Customer Dendogram")
dend=sc.dendrogram(sc.linkage(data,method='ward'))
plt.show()

#Create the cluster using the agglomerative algorithm
from sklearn.cluster import AgglomerativeClustering
cluster=AgglomerativeClustering(n_clusters=5,linkage='ward')
labels_=cluster.fit_predict(data)
print(labels_)

#for view the clusters in graph
plt.figure(figsize=(10,7))
plt.scatter(data.iloc[:,0],data.iloc[:,1],c=cluster.labels_,cmap='rainbow')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Hierarchical Clustering')
plt.show()


#performance metrics of the k means clustering
from sklearn.metrics import silhouette_score
silhoutte_coefficients=[]

for k in range(2,11):
    kmeans=AgglomerativeClustering(n_clusters=k,linkage='ward')
    kmeans.fit(data)
    score=silhouette_score(data,kmeans.labels_)
    silhoutte_coefficients.append(score)

print(silhoutte_coefficients)

plt.plot (range(2,11),silhoutte_coefficients)
plt.xticks(range(2,11))
plt.xlabel("number of clusters")
plt.ylabel("silhoutte_coefficients")
plt.show()