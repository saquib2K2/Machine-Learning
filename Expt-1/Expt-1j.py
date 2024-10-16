# Aim: Write a program to generate telecomm dataset which contains age, gender, usage minutes,churn status and draw pair plot of it.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# Seed for reproducibility
np.random.seed(42)
# Generate synthetic dataset
num_samples = 200
data = {
# Age between 18 and 70
'age': np.random.randint(18, 70, size=num_samples),
# Randomly choose gender
'gender': np.random.choice(['Male', 'Female'], size=num_samples),
# Usage with some normal distribution, clipped to non-negative values
'usage_minutes': np.random.normal(loc=300, scale=100,size=num_samples).clip(0),
# Randomly choose churn status
'churn_status': np.random.choice(['Churned', 'Not Churned'], size=num_samples)
}
df = pd.DataFrame(data)
# Encode categorical variables
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
df['churn_status'] = df['churn_status'].map({'Churned': 1, 'Not Churned': 0})
# Create pair plot
sns.pairplot(df, hue='churn_status', palette='viridis')
# Display plot
plt.show()
