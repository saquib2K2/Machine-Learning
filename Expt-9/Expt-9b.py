# Step 1: Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Step 2: Load the Iris dataset
iris = load_iris()
iris_data = iris.data
iris_target = iris.target
iris_target_names = iris.target_names
# Step 3: Standardize the data
scaler = StandardScaler()
iris_data_scaled = scaler.fit_transform(iris_data)
# Step 4: Perform PCA
pca = PCA(n_components=3)  # Reduce to 3 principal components for more features
iris_pca = pca.fit_transform(iris_data_scaled)
# Step 5: Visualize the cumulative explained variance
explained_variance = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance)
plt.figure(figsize=(8, 5))
plt.plot(range(1, 4), cumulative_explained_variance, marker='o', linestyle='--', color='b')
plt.title('Cumulative Explained Variance by Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid()
plt.show()
# Step 6: 2D Visualization of the first two principal components
plt.figure(figsize=(10, 7))
scatter = plt.scatter(iris_pca[:, 0], iris_pca[:, 1], c=iris_target, cmap='viridis', edgecolor='k', s=100)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D PCA of Iris Dataset')
# Modified legend creation:
# Manually create legend handles for each target class
handles = []
for i in range(len(iris_target_names)):
  handles.append(plt.scatter([], [], marker='o', s=100, edgecolor='k', c=[plt.cm.viridis(i / (len(iris_target_names) - 1))]))  
  # Use colormap to match scatter plot colors
plt.legend(handles, iris_target_names, title="Species") # Use the created handles
plt.grid()
plt.show()
# Step 7: 3D Visualization of the first three principal components
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(iris_pca[:, 0], iris_pca[:, 1], iris_pca[:, 2], c=iris_target, cmap='viridis', edgecolor='k', s=100)
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title('3D PCA of Iris Dataset')
# Modified legend creation (for 3D plot):
# Manually create legend handles for each target class
handles_3d = []
for i in range(len(iris_target_names)):
  handles_3d.append(ax.scatter([], [], [], marker='o', s=100, edgecolor='k', c=[plt.cm.viridis(i / (len(iris_target_names) - 1))]))
ax.legend(handles_3d, iris_target_names, title="Species") # Use the created handles for 3D plot
plt.show()
# Step 8: Print explained variance ratio for each principal component
print("Explained variance ratio for each principal component:", explained_variance)
