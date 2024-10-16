# To show the errors in prediction MAE, MSE etc
import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('placement.csv')
plt.scatter(df['cgpa'], df['package'])
plt.xlabel('CGPA')
plt.ylabel('Package(in lpa)')
plt.show()
X, Y = df.iloc[:, 0], df.iloc[:, 1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
lr_model.fit(np.array(X_train).reshape(-1, 1), Y_train)
root = tk.Tk()
root.title("Package (in LPA) Prediction")
root.geometry('800x600')
fig = Figure(figsize=(8, 6))
scatter_plot = fig.add_subplot(111)

def predict_package():
   try:
       cgpa = float(entry_cgpa.get())
       predicted_package = lr_model.predict([[cgpa]])
       label_prediction.config(text=f"Predicted Package: {predicted_package[0]:.2f} LPA")
       show_errors()
   except ValueError:
       label_prediction.config(text="Please enter a valid CGPA")

def show_errors():
   y_pred = lr_model.predict(np.array(X_test).reshape(-1, 1))
   mse = mean_squared_error(Y_test, y_pred)
   mae = mean_absolute_error(Y_test, y_pred)
   r2 = r2_score(Y_test, y_pred)
   rmse = np.sqrt(mse)
   error_text = f"Regression Metrics:- \n" \
                f"Mean Squared Error (MSE): {mse:.2f}\n" \
                f"Mean Absolute Error (MAE): {mae:.2f}\n" \
                f"R-squared (R2): {r2:.2f}\n" \
                f"Root Mean Squared Error (RMSE): {rmse:.2f}"
   label_errors.config(text=error_text)

label_cgpa = tk.Label(root, text="Enter CGPA:")
label_cgpa.pack(pady=10)
entry_cgpa = tk.Entry(root, width=10)
entry_cgpa.pack()
button_predict = tk.Button(root, text="Predict Package", command=predict_package)
button_predict.pack(pady=10)
label_prediction = tk.Label(root, text="")
label_prediction.pack(pady=10)
label_errors = tk.Label(root, text="")
label_errors.pack(pady=10)
scatter_plot.scatter(df['cgpa'], df['package'])
scatter_plot.plot(X_train, lr_model.predict(np.array(X_train).reshape(-1, 1)), color='orange')
scatter_plot.set_xlabel('CGPA')
scatter_plot.set_ylabel('Package')
scatter_plot.set_title('Package Prediction based on CGPA')
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
tk.mainloop()
