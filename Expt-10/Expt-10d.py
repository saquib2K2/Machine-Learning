# Step 1: Install the TabNet library if not already installed
# Uncomment the line below if you're using Google Colab
# !pip install pytorch-tabnet # Install pytorch-tabnet using pip
# Step 2: Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from pytorch_tabnet.tab_model import TabNetClassifier
# Step 3: Load the Adult Income dataset
# You can download the dataset from UCI Machine Learning Repository or use the following link
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
column_names = [
  "age", "workclass", "fnlwgt", "education", "education-num",
  "marital-status", "occupation", "relationship", "race", "sex",
  "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]
df = pd.read_csv(data_url, header=None, names=column_names, na_values=' ?', skipinitialspace=True)
# Display the first few rows of the dataset
print("\nFirst few rows of the dataset")
print(df.head())
# Step 4: Preprocess the data
# Drop rows with missing values
df.dropna(inplace=True)
# Encode categorical variables
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
  le = LabelEncoder()
  df[column] = le.fit_transform(df[column])
  label_encoders[column] = le
# Define features and target variable
X = df.drop('income', axis=1).values
y = df['income'].values
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Step 5: Train the TabNet model
tabnet_model = TabNetClassifier()
tabnet_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], max_epochs=100, patience=10)
# Step 6: Make predictions
y_pred = tabnet_model.predict(X_test)
# Step 7: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy of TabNet model: {accuracy * 100:.2f}%")
print("Classification Report")
print(classification_report(y_test, y_pred))
