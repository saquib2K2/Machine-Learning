# Aim: You are given a dataset containing information about a machine learning model's performance over multiple epochs during training. The dataset includes the loss and accuracy of the model for both the training and validation sets. Your task is to visualize this data using Matplotlib to better understand how the model's performance evolves over time and to identify potential issues such as overfitting.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Sample data
data = {
'epoch': np.arange(6, 25),
'train_loss': np.random.uniform(0.4, 0.1, 19),
'val_loss': np.random.uniform(0.8, 0.4, 19),
'train_accuracy': np.random.uniform(0.2, 0.65, 19),
'val_accuracy': np.random.uniform(0.4, 0.7, 19)
}
df = pd.DataFrame(data)
# Plotting Loss
plt.figure(figsize=(14, 6))
# Plot Training and Validation Loss
plt.subplot(1, 2, 1)
plt.plot(df['epoch'], df['train_loss'], label='Training Loss', color='blue',
marker='o')
plt.plot(df['epoch'], df['val_loss'], label='Validation Loss', color='red',
marker='o')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
# Plotting Accuracy
plt.subplot(1, 2, 2)
plt.plot(df['epoch'], df['train_accuracy'], label='Training Accuracy',
color='blue', marker='o')
plt.plot(df['epoch'], df['val_accuracy'], label='Validation Accuracy', color='red',
marker='o')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
# Show the plots
plt.tight_layout()
plt.show()
