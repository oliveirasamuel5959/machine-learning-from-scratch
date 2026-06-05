import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

def plot_data(X, y):
  # Create a scatter plot comparing Petal Length and Petal Width
  plt.figure(figsize=(10, 6))
  plt.scatter(X[:,2],X[:,3], c=y, cmap=cmap, edgecolor='k', s=20)
  plt.title("Iris Species: Sepal Dimensions")
  plt.xlabel("Petal Length (cm)")
  plt.ylabel("Petal Width (cm)")
  plt.show()