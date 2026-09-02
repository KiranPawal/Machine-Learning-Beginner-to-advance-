import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

df=pd.read_csv("D:\ML Course\Datasets\Mall_Customers.csv")
print(df.head())

input=df[['Age','Annual Income (k$)','Spending Score (1-100)']]
print(input.head())

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_scaled=scaler.fit_transform(input)
print(x_scaled)

#create dendogram using the scipy library
import scipy.cluster.hierarchy as sc
plt.figure(figsize=(10,7))
plt.title("Customer Dendogram")
dend=sc.dendrogram(sc.linkage(x_scaled,method='ward'))
plt.show()

#find require number of clusters 
from sklearn.cluster import KMeans
wcss=[]

for k in range(1,11):
    kmeans=KMeans(n_clusters=k,random_state=40,n_init=10)
    kmeans.fit(x_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss, marker='o')
plt.xlabel("Number of CLusters(k)")
plt.ylabel("WCSS/Inertia")
plt.title("Elbow Method")
plt.grid()
plt.show()

#create the cluster using this graph
#from sklearn.cluster import DBSCAN
#dbscan=DBSCAN(eps=0.5)

#value of k is k=3(number of clusters)
kmeans=KMeans(n_clusters=6,init='k-means++')
kmeans.fit(x_scaled)

kmeans.fit_predict(x_scaled)

# 6. Get cluster labels
df["Cluster"] = kmeans.labels_

print("\nData with cluster:")
print(df.head(10))

# PCA only for visualization
from sklearn.decomposition import PCA
pca = PCA(n_components=2)

x_pca = pca.fit_transform(x_scaled)

# Plot clusters
plt.figure(figsize=(8, 6))

plt.scatter(
    x_pca[:, 0],
    x_pca[:, 1],
    c=df["Cluster"],
    s=50
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("K-Means Clustering Using Multiple Inputs")
plt.show()