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

def plot_regression_line(y_test, y_pred, outdir=None):
  plt.figure(figsize=(10, 6))
  plt.scatter(y_test, y_pred, color="red", label="Predicted vs Actual")
  plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label="Ideal Fit")
  plt.title("KNN Regression: Predicted vs Actual")
  plt.xlabel("Actual Values")
  plt.ylabel("Predicted Values")
  
  if outdir:
    plt.savefig(outdir / "regression_line.png")
    print(f"Plot saved to: {outdir / 'regression_line.png'}")
    
  plt.legend()
  plt.show()
  
def plot_regression_results_all_features(X_train, y_train, X_test, y_test, y_pred, outdir=None):
  feature_names = [
    "Displacement (in³)",
    "Cylinders",
    "Horsepower (HP)",
    "Weight (lbs)",
    "Time to Acceleration from 0 to 60 mph (s)",
    "Model Year"
  ]

  fig, axes = plt.subplots(2, 3, figsize=(14, 8))
  axes = axes.flatten()

  for i, ax in enumerate(axes):
    # Training data
    ax.scatter(
      X_train[:, i],
      y_train,
      label="Train",
      alpha=0.6
    )

    # Actual test values
    ax.scatter(
      X_test[:, i],
      y_test,
      label="Test",
      alpha=0.6
    )

    # Predicted values
    ax.scatter(
      X_test[:, i],
      y_pred,
      marker="x",
      label="Prediction"
    )

    ax.set_title(feature_names[i])
    ax.set_xlabel(feature_names[i])
    ax.set_ylabel("Miles Per Gallon")

  # Single legend for entire figure
  handles, labels = axes[0].get_legend_handles_labels()
  fig.legend(handles, labels, loc="upper center", ncol=3)

  plt.tight_layout(rect=[0, 0, 1, 0.95])
  
  if outdir:
    plt.savefig(outdir / "regression_results.png")
    print(f"\n[OK] Plot saved to: {outdir / 'regression_results.png'}")
    
  plt.show()