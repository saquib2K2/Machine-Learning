import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
X = df.drop(['target'], axis='columns')
y = df.target
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create the GUI window
window = tk.Tk()
window.title("Iris Flower Classification")
window.geometry("400x400")
# Create input fields for features
labels = ["Sepal Length:", "Sepal Width:", "Petal Length:", "Petal Width:"]
entries = []
for i, label_text in enumerate(labels):
    label = ttk.Label(window, text=label_text)
    label.grid(row=i, column=0, padx=5, pady=5)
    entry = ttk.Entry(window)
    entry.grid(row=i, column=1, padx=5, pady=5)
    entries.append(entry)
# Create a dropdown menu for kernel selection
kernel_label = ttk.Label(window, text="Kernel:")
kernel_label.grid(row=4, column=0, padx=5, pady=5)
kernel_var = tk.StringVar(value="rbf")  # Default kernel
kernel_options = ["linear", "poly", "rbf", "sigmoid"]
kernel_dropdown = ttk.Combobox(window, textvariable=kernel_var, values=kernel_options)
kernel_dropdown.grid(row=4, column=1, padx=5, pady=5)
# Create a slider for regularization parameter (C)
c_label = ttk.Label(window, text="Regularization (C):")
c_label.grid(row=5, column=0, padx=5, pady=5)
c_var = tk.DoubleVar(value=1.0)  # Default C value
c_slider = ttk.Scale(window, from_=0.1, to=10.0, variable=c_var, orient="horizontal")
c_slider.grid(row=5, column=1, padx=5, pady=5)
# Create a button to predict
def predict():
    try:
        # Get input values and parameters
        input_values = [float(entry.get()) for entry in entries]
        kernel = kernel_var.get()
        c_value = c_var.get()
        # Create and train the model with selected parameters
        model = SVC(kernel=kernel, C=c_value)
        model.fit(X_train, y_train)
        # Make prediction
        prediction = model.predict([input_values])[0]
        flower_name = iris.target_names[prediction]
        # Display prediction
        result_label.config(text="Prediction: " + flower_name)
    except ValueError:
        messagebox.showerror("Input Error", "Invalid input. Please enter valid numbers for all features.")

# Create a button to predict
predict_button = ttk.Button(window, text="Predict", command=predict)
predict_button.grid(row=6, column=0, columnspan=2, padx=5, pady=10)

# Create a label to display the prediction
result_label = ttk.Label(window, text="")
result_label.grid(row=7, column=0, columnspan=2, padx=5, pady=5)

# Create a reset button to clear inputs
def reset():
    for entry in entries:
        entry.delete(0, tk.END)
    result_label.config(text="")
    kernel_var.set("rbf")
    c_var.set(1.0)
    c_slider.set(1.0)

reset_button = ttk.Button(window, text="Reset", command=reset)
reset_button.grid(row=8, column=0, columnspan=2, padx=5, pady=5)

# Start the GUI event loop
window.mainloop()
