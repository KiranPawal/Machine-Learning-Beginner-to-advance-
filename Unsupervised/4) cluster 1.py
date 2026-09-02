#A good real-world example is customer segmentation for an e-commerce company.
#The company has thousands of customers but does not know what type of customer each person is. 
# Instead of manually defining customer groups, we can use clustering to automatically discover groups with similar behavior.

#Clustering is an unsupervised learning technique because we don't provide the algorithm with predefined labels such as "Premium Customer" or "Occasional Customer".




import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

data = pd.DataFrame({
    "income": [3, 3.5, 4, 12, 15, 13, 7, 8],
    "purchases": [5, 7, 6, 35, 42, 38, 18, 20],
    "order_value": [800, 900, 850, 4500, 5200, 4800, 2000, 2200],
    "visits": [4, 5, 6, 25, 30, 28, 12, 14]
})

# Features
X = data[
    ["income", "purchases", "order_value", "visits"]
]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

# Train clustering model
kmeans.fit(X_scaled)

# Get cluster labels
data["cluster"] = kmeans.labels_

print(data)





import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.scatter(
    data["income"],
    data["purchases"],
    c=data["cluster"],
    cmap="viridis",
    s=100
)

# Plot cluster centers
centers = scaler.inverse_transform(kmeans.cluster_centers_)

plt.scatter(
    centers[:, 0],
    centers[:, 1],
    c="red",
    marker="X",
    s=200,
    label="Cluster Centers"
)

plt.xlabel("Income")
plt.ylabel("Purchases")
plt.title("K-Means Customer Clusters")
plt.legend()
plt.grid(True)

plt.show()