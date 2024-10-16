#Program_1:
# Step 1: Install XGBoost if not already installed
# You may need to run this in Google Colab
# !pip install xgboost
# Step 2: Import the necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
# Import matplotlib.pyplot
import matplotlib.pyplot as plt # This line is added to import the pyplot module
# Step 3: Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Step 4: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Step 5: Create the XGBoost model
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
# Step 6: Train the model
xgb_model.fit(X_train, y_train)
# Step 7: Make predictions
y_pred = xgb_model.predict(X_test)
# Step 8: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of XGBoost model: {accuracy * 100:.2f}%")
print("\nClassification Report")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
# Step 9: Feature Importance Plot
xgb.plot_importance(xgb_model)
plt.title('Feature Importance of XGBoost Model') # plt is now defined and can be used
plt.show()