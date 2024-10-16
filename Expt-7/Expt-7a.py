import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
# Load Iris dataset
iris = load_iris()
# Create DataFrame from the Iris dataset
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['flower_name'] = df.target.apply(lambda x: iris.target_names[x])
df0 = df[:50]
df1 = df[50:100]
df2 = df[100:]
# Plot Sepal Length vs Sepal Width for different classes
plt.title("Plot Sepal Length vs Sepal Width")
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'], color="green", marker='+', label="Setosa")
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'], color="blue", marker='.', label="Versicolor")
plt.legend()
plt.show()
# Plot Petal Length vs Petal Width for different classes
plt.title("Plot Petal Length vs Petal Width")
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.scatter(df0['petal length (cm)'], df0['petal width (cm)'], color="green", marker='+', label="Setosa")
plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'], color="blue", marker='.', label="Versicolor")
plt.legend()
plt.show()

X = df.drop(['target', 'flower_name'], axis='columns')
y = df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train SVM model (default parameters)
model = SVC()
model.fit(X_train, y_train)
model.predict([[4.8, 3.0, 1.5, 0.3]])
model_C_1 = SVC(C=1)
model_C_1.fit(X_train, y_train)
model_C_10 = SVC(C=10)
model_C_10.fit(X_train, y_train)
model_g_10 = SVC(gamma=10)
model_g_10.fit(X_train, y_train)
model_linear = SVC(kernel='linear')
model_linear.fit(X_train, y_train)
