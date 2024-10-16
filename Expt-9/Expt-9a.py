# Step 1: Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# Step 2: Load the Iris dataset
iris = load_iris()
iris_data = iris.data
iris_target = iris.target
iris_target_names = iris.target_names
# Step 3: Standardize the data
scaler = StandardScaler()
iris_data_scaled = scaler.fit_transform(iris_data)
# Step 4: Perform PCA
pca = PCA(n_components=2)  # Reduce to 2 principal components for visualization
iris_pca = pca.fit_transform(iris_data_scaled)
# Step 5: Visualize the results
plt.figure(figsize=(8, 6))
scatter = plt.scatter(iris_pca[:, 0], iris_pca[:, 1], c=iris_target, cmap='viridis', edgecolor='k', s=100)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Iris Dataset')
# Modified legend creation
# Instead of using scatter.legend_elements(), manually create the legend handles
# using the unique target values and corresponding labels.
for target, target_name in zip(np.unique(iris_target), iris_target_names):
    plt.scatter([], [], c=[plt.cm.viridis(target / 2)], label=target_name, edgecolor='k', s=100)
plt.legend(title="Species")
plt.grid()
plt.show()
# Optional: View the explained variance ratio
print("\nExplained variance ratio:", pca.explained_variance_ratio_)   
