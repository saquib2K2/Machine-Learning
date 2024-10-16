import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load your dataset into x
# Assuming you are working with the 'Mall_Customers.csv' dataset
df = pd.read_csv("Mall_Customers.csv")
x = df[['Annual Income (k$)', 'Spending Score (1-100)']]  # Select features for clustering

# Calculate WCSS for different numbers of clusters
wcss = []
for k in range(1, 11):
    k_means = KMeans(n_clusters=k, random_state=42)
    k_means.fit(x)
    wcss.append(k_means.inertia_)

# Plot the results using the Elbow Method
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
