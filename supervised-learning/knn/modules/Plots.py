import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from mlxtend.plotting import plot_decision_regions 
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# ----------------------------------------------
# Visualization Functions for KNN Classification
# ----------------------------------------------
def plot_data(data, cols, title):
  plt.figure(figsize=(10, 6))

  for species in data['target'].unique():
    subset = data[data['target'] == species]

    plt.scatter(
      subset[cols[0]],
      subset[cols[1]],
      edgecolor='k',
      s=30,
      label=species
    )

  plt.title(title)
  plt.xlabel(cols[0])
  plt.ylabel(cols[1])
  plt.legend()
  plt.grid(True, alpha=0.3)
  plt.show()

# ----------------------------------------------

def plot_confusion_matrix(cm, classes, outdir=None):
  plt.figure(figsize=(8, 6))
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
  plt.xlabel('Predicted Label')
  plt.ylabel('True Label')
  plt.title('Confusion Matrix')
  
  if outdir:
    plt.savefig(outdir / "confusion_matrix.png")
    print(f"Plot saved to: {outdir / 'confusion_matrix.png'}")
    
  plt.show()

# ----------------------------------------------
  
def plot_inference(data: pd.DataFrame, cols: list, pred: int):
  plt.figure(figsize=(10, 6))
  plt.scatter(data[cols[0]], data[cols[1]], c=data['target'], cmap=cmap, edgecolor='k', s=20, label=data['target'])
  plt.title(f'Inference Result: Predicted Class {pred}')
  plt.xlabel(cols[0])
  plt.ylabel(cols[1])
  plt.show()

# ----------------------------------------------
  
def plot_decision_boundary(X, y, model):
  plot_decision_regions(X=X, y=y, clf=model, legend=2)
  plt.xlabel('Feature 1')
  plt.ylabel('Feature 2')
  plt.title('KNN Decision Boundary')
  plt.show()

# ----------------------------------------------
# Visualization Functions for KNN Regression
# ----------------------------------------------
def plot_original_data(X, y, outdir=None):
  plt.figure(figsize=(10, 6))
  plt.scatter(X[:, 0], y, label="Original Data", color="blue")
  plt.title("Original Data")
  plt.xlabel("X Values")
  plt.ylabel("Y Values")
  
  if outdir:
    plt.savefig(outdir / "original_data.png")
    print(f"Plot saved to: {outdir / 'original_data.png'}")
    
  plt.legend()
  plt.show()
  
def plot_regression_results(X_train, y_train, X_test, y_test, y_pred, outdir=None):
  plt.figure(figsize=(10, 6))
  
  # Plot the training data points
  plt.scatter(X_train[:, 3], y_train, label="Training Data", color="blue")
  # Plot the test data points and their predictions
  plt.scatter(X_test[:, 3], y_test, label="Test Predictions", color="green", marker="o")
  # Plot the predicted values for the test set
  plt.scatter(X_test[:, 3], y_pred, label="Predicted Values", color="red", marker="x")
  
  plt.title(f"KNN Regression Results")
  plt.xlabel("Acceleration")
  plt.ylabel("Fuel Consumption")
  plt.legend()
  
  if outdir:
    plt.savefig(outdir / "knn_regression_results.png")
    print(f"Plot saved to: {outdir / 'knn_regression_results.png'}")
  plt.show()