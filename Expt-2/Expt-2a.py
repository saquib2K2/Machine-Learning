import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import tkinter as tk
df = pd.read_csv("placement.csv")
root = tk.Tk()
root.title("CGPA vs Package Prediction")
root.geometry("800x600")
fig = Figure(figsize=(8, 6))
scatter_plot = fig.add_subplot(111)
x = df.iloc[:, 0].values.reshape(-1, 1)
y = df.iloc[:, -1].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
model = LinearRegression()
model.fit(x_train, y_train)
model.score(x_test, y_test)

def predict_package():
   try:
       cgpa = float(entry_cgpa.get())
       predicted_package = model.predict([[cgpa]])
       label_prediction.config(text=f"Predicted Package: {predicted_package[0]:.2f} LPA")
   except ValueError:
       label_prediction.config(text="Please enter a valid CGPA")

label_cgpa = tk.Label(root, text="Enter CGPA:")
label_cgpa.pack(pady=10)
entry_cgpa = tk.Entry(root, width=10)
entry_cgpa.pack()
button_predict = tk.Button(root, text="Predict Package", command=predict_package)
button_predict.pack(pady=10)
label_prediction = tk.Label(root, text="")
label_prediction.pack(pady=10)
scatter_plot.scatter(df['cgpa'], df['package'])
# Use x_train and predict on it
scatter_plot.plot(x_train, model.predict(x_train), color='red')  
scatter_plot.set_xlabel('CGPA')
scatter_plot.set_ylabel('Package')
scatter_plot.set_title('CGPA vs Package ')
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

tk.mainloop()
