#Program_3:
# Step 1: Install LightGBM if not already installed
# Uncomment the line below if you're using Google Colab
# !pip install lightgbm
# Step 2: Import necessary libraries
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import lightgbm as lgb
# Step 3: Load the Breast Cancer dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)
# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Step 5: Create the LightGBM model
lgb_model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1)
# Step 6: Train the model
lgb_model.fit(X_train, y_train)
# Step 7: Make predictions
y_pred = lgb_model.predict(X_test)
# Step 8: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy of LightGBM model: {accuracy * 100:.2f}%")
print("\nClassification Report")
print(classification_report(y_test, y_pred, target_names=data.target_names))
# Optional: Feature Importance Plot
lgb.plot_importance(lgb_model, max_num_features=10, importance_type='split', title='Feature Importance ')
plt.show()