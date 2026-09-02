import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

df=pd.read_csv("D:\ML Course\Datasets\Mall_Customers.csv")
print(df.head())

print(df.isnull().sum().sum())

x=df[["Age","Annual Income (k$)","Spending Score (1-100)"]]
print(x.head())
print(x.shape)


#from below this i have use the Gaussian Distribution Anomaly Detection algorithm
mean = x.mean()            ## Calculate mean
std = x.std()            ## Calculate standard deviation

print("Mean:", mean)
print("Standard Deviation:", std)

# Calculate Gaussian probability
probability_each_feature = norm.pdf(x, mean, std)

print("\nProbability for each feature:")
print(probability_each_feature)

print("\nShape:")
print(probability_each_feature.shape)

# --------------------------------------------------
# 5. Combine probabilities
# --------------------------------------------------
probability = probability_each_feature.prod(axis=1)

print("\nCombined probability:")
print(probability)

print("\nProbability shape:")
print(probability.shape)

#add probability to dataframe
df["Probability"] = probability


# 7. Set anomaly threshold
threshold = np.percentile(probability, 5)

print("\nProbability threshold:")
print(threshold)

# 8. Detect anomalies
df["Anomaly"] = np.where(
    df["Probability"] < threshold,           #-1 = Anomaly    1=normal
    -1,
    1
)

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
plt.title("Gaussian Distribution Anomaly Detection")
plt.legend()

plt.show()
