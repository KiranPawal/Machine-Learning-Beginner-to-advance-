import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()
print(cancer.keys())

print(cancer.DESCR) 

df=pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
print(df.head())

#before use the PCA technique need to use the scaling on data
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(df)

scaled_data=scaler.transform(df)
print(scaled_data)

#use the PCA technique 
from sklearn.decomposition import PCA
PCA=PCA(n_components=2)
PCA.fit(scaled_data)

x_pca=PCA.transform(scaled_data)
print(scaled_data.shape)
print(x_pca.shape)

#view the this lower diamention data
print(x_pca)

#plot the data base on this target value 
plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'])
plt.xlabel('FIRST PRINCIPLE COMPONENT')
plt.ylabel('SECOND PRINCIPLE COMPONENT')
plt.show()