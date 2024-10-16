#Program_2:
# Step 1: Install CatBoost if not already installed
# Uncomment the line below if you're using Google Colab
#!pip install catboost # Install the catboost package
# Step 2: Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor
# Step 3: Load a house prices dataset
# For this example, we'll use a simplified dataset
# You can replace this with any dataset that you have
data_url = 'https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv'
df = pd.read_csv(data_url)
# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())
# Step 4: Preprocess the data
# Split the data into features and target variable
X = df.drop(['medv'], axis=1)  # 'medv' is the median value of owner-occupied homes (target)
y = df['medv']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Step 5: Initialize the CatBoostRegressor
catboost_model = CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=6, verbose=100)
# Step 6: Train the model
catboost_model.fit(X_train, y_train)
# Step 7: Make predictions
y_pred = catboost_model.predict(X_test)
# Step 8: Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
print(f"\nMean Absolute Error of the CatBoost model: {mae:.2f}")
# Optional: Feature Importance Plot
feature_importances = catboost_model.get_feature_importance(prettified=True)
print("\nFeature Importance:")
print(feature_importances)