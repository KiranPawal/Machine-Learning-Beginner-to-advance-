import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("D:\ML Course\Datasets\Mall_Customers.csv")
print(df.head())

print(df.isnull().sum().sum())

x=df[["Age","Annual Income (k$)","Spending Score (1-100)"]]
print(x.head())
print(x.shape)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
print("Scaled data")
print(x_scaled)

# ----------------------------------------------------
# 6. Create Isolation Forest model
# ----------------------------------------------------
from sklearn.ensemble import IsolationForest
model = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)
df["anamaly"]=model.fit_predict(x_scaled)                          #Train model and predict anomalies
df["Anomaly_Score"] = model.decision_function(x_scaled)


print("\nData with anomaly results:")               ## 9. Display results
print(df.head(10))

print("\nAnomaly count:")
print(df["anamaly"].value_counts())                                 

# 8. Display only anomalies
anomalies = df[df["anamaly"]==-1]

print("Anomalies:")
print(anomalies)

