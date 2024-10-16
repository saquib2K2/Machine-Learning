import pandas as pd
import numpy as np
# Generate synthetic data
np.random.seed(0)
n = 1000 # Number of records
# Generating random data for each column
age = np.random.randint(18, 80, size=n)
gender = np.random.choice(['Male', 'Female'], size=n)
usage_minutes = np.random.randint(50, 200, size=n)
data_usage = np.random.uniform(1.0, 10.0, size=n)
churn_status = np.random.choice(['Churned', 'Not Churned'], size=n, p=[0.2, 0.8])
# Convert gender to 0 and 1
gender_numeric = np.where(gender == 'Male', 0, 1)
# Convert churn_status to 0 and 1
churn_status_numeric = np.where(churn_status == 'Churned', 1, 0)
# Create DataFrame
df = pd.DataFrame({
'age': age,
'gender': gender_numeric,
'usage_minutes': usage_minutes,
'data_usage': data_usage,
'churn_status': churn_status_numeric
})
# Save DataFrame to CSV
df.to_csv('telecom_data.csv', index=False)
# Display the first few rows of the generated dataset
print(df.head())
