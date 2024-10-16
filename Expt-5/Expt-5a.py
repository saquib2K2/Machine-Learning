# Importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
# Reading the csv file and putting it into 'df' object.
df = pd.read_csv('adult_dataset.csv')
df.info()
df.head()
df.isin(['?']).sum(axis=0)
df=df.replace('?',np.nan)
df=df.dropna()
df.isin(['?']).sum(axis=0)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df=df.apply(le.fit_transform)
df.head()
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score
x=df.drop('income',axis=1)
y=df['income']
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
clf=DecisionTreeClassifier(max_depth=3)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)
plt.figure(figsize=(10,8))
plot_tree(decision_tree=clf,feature_names=x.columns,class_names=['<=50k','>50k'],filled=True)
plt.show()
