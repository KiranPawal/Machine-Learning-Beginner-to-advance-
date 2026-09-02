import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X,y=make_blobs(n_samples=300,centers=3,cluster_std=1.0,random_state=42)
print(X)

#how to find the number of centres
wcss=[]
# Generate WCSS for 10 clusters
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

print(wcss)
print(len(wcss))

#plot the graph of wcss and no. of clusters
plt.plot(range(1,11),wcss)
plt.xticks(range(1,11))
plt.xlabel("number of clusters")
plt.ylabel("WCSS")
plt.show()

#find the value of k
from kneed import KneeLocator
k1 = KneeLocator(
    range(1, 11),
    wcss,
    curve="convex",
    direction="decreasing"
)
print(k1.elbow)

# train the model
from sklearn.mixture import GaussianMixture
gmm=GaussianMixture(n_components=3)
gmm.fit(X)

labels=gmm.predict(X)

plt.scatter(X[:,0],X[:,1],c=labels)
plt.title("Gaussian Mixture model")
plt.show() 