import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from mlxtend.plotting import plot_decision_regions 
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

def plot_data(data: pd.DataFrame, cols:list, title="Iris Dataset: Feature 1 vs Feature 2"):
  # Create a scatter plot comparing sepal length and sepal width
  plt.figure(figsize=(10, 6))
  plt.scatter(data[cols[0]], data[cols[1]], c=data['target'], cmap=cmap, edgecolor='k', s=20, label=data['target'])
  plt.title(title)
  plt.xlabel(cols[0])
  plt.ylabel(cols[1])
  plt.show()
  
def plot_inference(data: pd.DataFrame, cols: list, pred: int):
  plt.figure(figsize=(10, 6))
  plt.scatter(data[cols[0]], data[cols[1]], c=data['target'], cmap=cmap, edgecolor='k', s=20, label=data['target'])
  plt.title(f'Inference Result: Predicted Class {pred}')
  plt.xlabel(cols[0])
  plt.ylabel(cols[1])
  plt.show()
  
def plot_decision_boundary(X, y, model):
  plot_decision_regions(X=X, y=y, clf=model, legend=2)
  plt.xlabel('Feature 1')
  plt.ylabel('Feature 2')
  plt.title('KNN Decision Boundary')
  plt.show()
