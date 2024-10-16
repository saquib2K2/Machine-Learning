import pandas as pd
from tkinter import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.cluster import KMeans

df = pd.read_csv("Mall_Customers.csv")
# Prepare the data for clustering
x = df[['Annual Income (k$)', 'Spending Score (1-100)']]

k_mean = KMeans(n_clusters=5, random_state=42)
y_mean = k_mean.fit_predict(x)

def show_entry_fields():
  p1 = int(e1.get())
  p2 = int(e2.get())
  
  result = k_mean.predict([[p1, p2]])
  print("This customer belongs to cluster no:", result[0])

  cluster_info = {
      0: "customer with medium annual income & medium annual spending score",
      1: "customer with high annual income & low annual spending score",
      2: "customer with low annual income & low annual spending score",
      3: "customer with low annual income & high annual spending score",
      4: "customer with high annual income & high annual spending score"
  }

  for widget in master.grid_slaves():
    if int(widget.grid_info()["row"]) >= 4:  # Assuming info labels start from row 4
      widget.destroy()
  
  Label(master, text=cluster_info[result[0]]).grid(row=4)

master = Tk()
master.title("Customer Segmentation using Machine Learning")

Label(master, text="Customer Segmentation using Machine Learning", bg="Yellow", fg="black").grid(row=0, columnspan=2)
# Input labels and entrie
Label(master, text="Annual Income").grid(row=1)
Label(master, text="Spending Score").grid(row=2)
e1 = Entry(master)
e2 = Entry(master)
e1.grid(row=1, column=1)
e2.grid(row=2, column=1)

Button(master, text='Predict', command=show_entry_fields).grid(row=3)

figure3 = plt.Figure(figsize=(5, 4), dpi=100)
ax3 = figure3.add_subplot(111)

for i in range(5):  # Since n_clusters=5
  ax3.scatter(x.iloc[y_mean == i, 0], x.iloc[y_mean == i, 1], s=100, label=f'Cluster {i}')

ax3.set_xlabel('Annual Income (k$)')
ax3.set_ylabel('Spending Score (1-100)')
ax3.set_title('Annual Income vs Spending Score')
ax3.legend()

scatter3 = FigureCanvasTkAgg(figure3, master)
scatter3.get_tk_widget().grid(row=5, columnspan=2)
master.mainloop()
