import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
digits=load_digits()
x=digits.data
y=digits.target

print(x.shape)

#It can convert in dataframe 
df=pd.DataFrame(x)
print(df.head())

#apply feature scaling on data 
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_scaled=scaler.fit_transform(df)

#apply the SVD algorithm on data 
from sklearn.decomposition import TruncatedSVD
svd=TruncatedSVD(n_components=10)            #n_components=10   i.e, data ko kitane feature me reduce karna hai
x_svd=svd.fit_transform(x_scaled)

#view the reduce data 
print("Reduces shape:",x_svd.shape)
print(x_svd)

#variance: it shows the how much information is preserved the reduce and SVD converted data
variance=svd.explained_variance_ratio_
print(variance)

#total information preserve 
print("total information preserved:",variance.sum())

# i have draw the graph of the model information preserve taking value of n_component=1 to 11

plt.plot(
    range(1,11),
    variance.cumsum(),
    marker='o'
)
plt.xlabel("Number of Components")
plt.ylabel("Cummulative explained varience")
plt.title("Varience preserved by SVD")
plt.show()