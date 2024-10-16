import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Load the dataset
df = pd.read_csv('telecom_data.csv')
# Basic pairplot to visualize relationships
sns.pairplot(df, hue='churn_status', palette='Set1')
plt.title('Pairplot of Telecom Customer Attributes by Churn Status')
plt.show()
# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Telecom Customer Attributes')
plt.show()
# Boxplot of age vs. churn_status
plt.figure(figsize=(8, 6))
sns.boxplot(x='churn_status', y='age', data=df, palette='Set2')
plt.title('Boxplot of Age by Churn Status')
plt.show()
# Countplot of churn_status
plt.figure(figsize=(6, 5))
sns.countplot(x='churn_status', data=df, palette='Set3')
plt.title('Count of Churn Status')
plt.show()
