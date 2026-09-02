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


# --------------------------------------------------
# 6. Create LOF model
# --------------------------------------------------
from sklearn.svm import OneClassSVM

model = OneClassSVM(
    kernel="rbf",
    nu=0.05
)

df["Anomaly"] = model.fit_predict(x_scaled)


# --------------------------------------------------
# 8. Calculate anomaly scores
# --------------------------------------------------
df["Anomaly_Score"] = model.decision_function(x_scaled)



# --------------------------------------------------
# 9. Display results
# --------------------------------------------------
print("\nData with anomaly results:")
print(df.head())

# --------------------------------------------------
# 10. Count normal and anomalous records
# --------------------------------------------------
print("\nAnomaly count:")
print(df["Anomaly"].value_counts())

# --------------------------------------------------
# 11. Display only anomalies
# --------------------------------------------------
anomalies = df[df["Anomaly"] == -1]

print("\nDetected anomalies:")
print(anomalies)


# --------------------------------------------------
# 13. Save results
# --------------------------------------------------
df.to_csv(
    "customers_lof_results.csv",
    index=False
)

print("\nResults saved to customers_lof_results.csv")


# --------------------------------------------------
# 14. Visualization
# --------------------------------------------------
normal = df[df["Anomaly"] == 1]
anomalies = df[df["Anomaly"] == -1]

plt.figure(figsize=(10, 6))

plt.scatter(
    normal["Annual Income (k$)"],
    normal["Spending Score (1-100)"],
    s=60,
    label="Normal"
)

plt.scatter(
    anomalies["Annual Income (k$)"],
    anomalies["Spending Score (1-100)"],
    s=100,
    marker="x",
    label="Anomaly"
)

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Local Outlier Factor (LOF) Anomaly Detection")
plt.legend()

plt.show()
