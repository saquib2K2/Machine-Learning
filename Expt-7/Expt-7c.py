import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the Diabetes dataset
diabetes = load_diabetes()
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target

# Select fewer features correctly
X = df[['age', 'bmi', 'bp', 's5']]  # 'bp' is blood pressure and 's5' is a selected feature
y = df['target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the GUI window
window = tk.Tk()
window.title("Diabetes Progression Prediction")
window.geometry("400x400")

# Create input fields for selected features
labels = {
    'age': "Age (years)",
    'bmi': "Body Mass Index (BMI)",
    'bp': "Blood Pressure (mm Hg)",
    's5': "S5 Measurement"
}

entries = []
for i, (key, label_text) in enumerate(labels.items()):
    label = ttk.Label(window, text=label_text + ":")
    label.grid(row=i, column=0, padx=5, pady=5)
    entry = ttk.Entry(window)
    entry.grid(row=i, column=1, padx=5, pady=5)
    entries.append(entry)

# Create a button to predict
def predict():
    try:
        # Get input values
        input_values = [float(entry.get()) for entry in entries]
        # Create and train the model
        model = LinearRegression()
        model.fit(X_train, y_train)
        # Make prediction
        prediction = model.predict([input_values])[0]
        # Display prediction
        result_label.config(text="Disease Progression: {:.2f}".format(prediction))
    except ValueError:
        messagebox.showerror("Input Error", "Invalid input. Please enter valid numbers for all features.")

# Create a button to predict
predict_button = ttk.Button(window, text="Predict", command=predict)
predict_button.grid(row=len(labels), column=0, columnspan=2, padx=5, pady=10)

# Create a label to display the prediction
result_label = ttk.Label(window, text="")
result_label.grid(row=len(labels) + 1, column=0, columnspan=2, padx=5, pady=5)

# Create a reset button to clear inputs
def reset():
    for entry in entries:
        entry.delete(0, tk.END)
    result_label.config(text="")

reset_button = ttk.Button(window, text="Reset", command=reset)
reset_button.grid(row=len(labels) + 2, column=0, columnspan=2, padx=5, pady=5)

# Start the GUI event loop
window.mainloop()
