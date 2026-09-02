import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# create dataset using make_blob
X,Y=make_blobs(n_samples=1000,n_features=2,centers=3,random_state=23)
print(X.shape)

#plot the this data 
plt.scatter(X[:,0],X[:,1],c=Y)
plt.show()

#split data into training and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=40)

#apply the k_means algorithm
from sklearn.cluster import KMeans

#but i dont know the value of k or number of clusters so 
#two types to find the value of clusters   1.) manually  and 2)automaticaly using the kneed library

#  1) manually
wcss=[]
for k in range (1,11):
    kmeans=KMeans(n_clusters=k,init='k-means++')
    kmeans.fit(x_train)
    wcss.append(kmeans.inertia_)
print(wcss)

#plot the graph of wcss and no. of clusters
plt.plot(range(1,11),wcss)
plt.xticks(range(1,11))
plt.xlabel("number of clusters")
plt.ylabel("WCSS")
plt.show()

#value of k is k=3(number of clusters)
kmeans=KMeans(n_clusters=3,init='k-means++')
kmeans.fit(x_train)

#find the label of x_train
y_label=kmeans.fit_predict(x_train)

#graph of the x_train
plt.scatter(x_train[:,0],x_train[:,1],c=y_label)
plt.show()

#label of x_test
y_test_label=kmeans.fit_predict(x_test)

#graph of x_test
plt.scatter(x_test[:,0],x_test[:,1],c=y_test_label)
plt.show()


#  2)automatic find the value of k
from kneed import KneeLocator
kl=KneeLocator(range(1,11),wcss,curve="convex",direction="decreasing")
print(kl.elbow)

#performance metrics of the k means clustering
from sklearn.metrics import silhouette_score
silhoutte_coefficients=[]

for k in range(2,11):
    kmeans=KMeans(n_clusters=k,init='k-means++')
    kmeans.fit(x_train)
    score=silhouette_score(x_train,kmeans.labels_)
    silhoutte_coefficients.append(score)

print(silhoutte_coefficients)

plt.plot (range(2,11),silhoutte_coefficients)
plt.xticks(range(2,11))
plt.xlabel("number of clusters")
plt.ylabel("silhoutte_coefficients")
plt.show()