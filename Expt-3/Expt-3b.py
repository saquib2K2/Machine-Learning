import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import tkinter as tk
from tkinter import ttk

data = pd.read_csv('add.csv')
data.head()
x = data[['x', 'y']]
y = data['sum']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)
model.score(x_train, y_train)
model.score(x_test, y_test)
y_pred = model.predict(x_test)

class AdditionApp:
   def __init__(self, root):
       self.root = root
       self.root.title("Addition Using Multiple Linear Regression")
       self.create_widgets()

   def create_widgets(self):
       ttk.Label(self.root, text="Number 1:").grid(column=0, row=0, padx=10, pady=10)
       self.num1_var = tk.DoubleVar()
       self.num1_entry = ttk.Entry(self.root, textvariable=self.num1_var)
       self.num1_entry.grid(column=1, row=0, padx=10, pady=10)
       ttk.Label(self.root, text="Number 2:").grid(column=0, row=1, padx=10, pady=10)
       self.num2_var = tk.DoubleVar()
       self.num2_entry = ttk.Entry(self.root, textvariable=self.num2_var)
       self.num2_entry.grid(column=1, row=1, padx=10, pady=10)
       self.calc_button = ttk.Button(self.root, text="Calculate", command=self.calculate_sum)
       self.calc_button.grid(column=0, row=2, columnspan=2, pady=10)
       self.result_var = tk.StringVar()
       ttk.Label(self.root, text="Result:").grid(column=0, row=3, padx=10, pady=10)
       self.result_label = ttk.Label(self.root, textvariable=self.result_var)
       self.result_label.grid(column=1, row=3, padx=10, pady=10)

   def calculate_sum(self):
       num1 = self.num1_var.get()
       num2 = self.num2_var.get()
       prediction = model.predict([[num1, num2]])[0]
       self.result_var.set(f"{prediction:.2f}")

root = tk.Tk()
app = AdditionApp(root)
root.mainloop()