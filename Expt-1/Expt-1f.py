""" PROGRAM:
Aim: You have a dataset of customer information and their purchasing behavior. The dataset includes columns such as customer_id, age, gender, annual_income, purchase_amount, and purchase_date. You need to preprocess this data to prepare it for machine learning tasks. The preprocessing steps include: Handling missing values. Encoding categorical variables. Normalizing numerical variables. Creating a new feature: the total purchase amount for each customer."""
import pandas as pd
from sklearn.preprocessing import StandardScaler
# Sample DataFrame creation
data = {
'customer_id': [1, 2, 3, 4, 5],
'age': [25, 30, None, 45, 50],
'gender': ['Male', 'Female', 'Female', None, 'Male'],
'annual_income': [50000, 60000, 70000, 80000, None],
'purchase_amount': [150, None, 200, 300, 400],
'purchase_date':['2024-01-01','2024-01-02','2024-01-03','2024-01-04','2024-01-05']
}
df = pd.DataFrame(data)
print(df)

df = pd.DataFrame(data)
# Handling Missing Values
df['age'].fillna(df['age'].mean(), inplace=True)
df['gender'].fillna(df['gender'].mode()[0], inplace=True)
df['purchase_amount'].fillna(df['purchase_amount'].median(), inplace=True)
df['annual_income'].fillna(df['annual_income'].median(), inplace=True)
# Encoding Categorical Variables
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
# Normalizing Numerical Variables
scaler = StandardScaler()
df[['age', 'annual_income', 'purchase_amount']] = scaler.fit_transform(df[['age',
'annual_income', 'purchase_amount']])
# Creating a New Feature
# Assuming each record is unique per customer for simplicity
df['total_purchase_amount'] = df['purchase_amount']
# Drop the purchase_date column if not needed for ML tasks
df.drop(columns=['purchase_date'], inplace=True)
print(df)

