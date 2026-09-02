import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

df = pd.read_csv("D:/ML Course/Datasets/bank_transactions_data_2.csv")
print(df.head())

print(df.columns)


#i have Minus the transaction date from the previous transaction date for find the time difference
# timedifference = previous transaction date - transaction date
df["TransactionDate"] = pd.to_datetime(
    df["TransactionDate"]
)
df["PreviousTransactionDate"] = pd.to_datetime(
    df["PreviousTransactionDate"]
)

df["TimeDifference"] = (
    df["TransactionDate"]
    - df["PreviousTransactionDate"]
)
print(df["TimeDifference"].head())

#create the new dataset
x=df[["TransactionAmount","TransactionType","Channel","TransactionDuration","LoginAttempts","AccountBalance","TimeDifference"]]


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
x["TransactionType"] = encoder.fit_transform(
    x["TransactionType"]
)
x["Channel"] = encoder.fit_transform(
    x["Channel"]
)

#remove the time from the TimeDifference column only day difference is remaining
x["TimeDifference"] = x["TimeDifference"].dt.days
print(x.head())

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_scaled = scaler.fit_transform(x)

print("scaled data")
print(x_scaled)

#anamaly detection using clustring algorithm
from sklearn.cluster import DBSCAN
model = DBSCAN(eps=0.5, min_samples=5)

from sklearn.svm import OneClassSVM
model = OneClassSVM(
    kernel="rbf",
    nu=0.05
)


df["Anomaly"]= model.fit_predict(x_scaled)


print("\nAnomaly count:")
print(df["Anomaly"].value_counts())
print(df.shape)

anomalies = df[df["Anomaly"] != -1]
print("\nDetected anomalies:")
print(anomalies)



normal = df[df["Anomaly"] == 1]
anomalies = df[df["Anomaly"] == -1]

plt.figure(figsize=(10, 6))

plt.scatter(
    normal["TransactionAmount"],
    normal["AccountBalance"],
    s=60,
    label="Normal"
)

plt.scatter(
    anomalies["TransactionAmount"],
    anomalies["AccountBalance"],
    s=100,
    marker="x",
    label="Anomaly"
)

plt.xlabel("TransactionAmount")
plt.ylabel("AccountBalance")
plt.title("Local Outlier Factor (LOF) Anomaly Detection")
plt.legend()

plt.show()
