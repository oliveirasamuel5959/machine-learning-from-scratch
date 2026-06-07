import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo 

from modules.Plots import plot_confusion_matrix, plot_data, plot_decision_boundary, plot_inference
from modules.KNN import KNN

ROOT_DIR = Path(__file__).parent.parent.parent
OUTPUT_DIR = ROOT_DIR / 'outputs-results' / 'knn_classification'

print(f"Current working directory: {ROOT_DIR}")

def main():
  # -------------------------------------------
  # Create output directory if it doesn't exist
  # -------------------------------------------
  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
  print(f"Output directory: {OUTPUT_DIR}")
  
  # -------------
  # Load Dataset
  # -------------
  data = fetch_ucirepo('iris')
  X = data.data.features
  y = data.data.targets

  # ------------------
  # Original dataset
  # ------------------
  df = pd.DataFrame(X, columns=data.data.feature_names)
  df['target'] = y
  
  print(df.head())

  # ------------------------------------------
  # Split the data into training and test sets
  # ------------------------------------------
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  
  X_train = np.array(X_train)
  X_test = np.array(X_test)
  y_train = np.array(y_train)
  y_test = np.array(y_test)
  
  # --------------------------
  # Encode the target labels
  # --------------------------
  le = LabelEncoder()
  y_train = le.fit_transform(y_train)
  y_test = le.transform(y_test)
  
  print(f'Training set size: {X_train.shape} and {y_train.shape}')
  print(f'Training set class distribution: {np.bincount(y_train)}')
  print(f'Test set size: {X_test.shape} and {y_test.shape}')
  print(f'Test set class distribution: {np.bincount(y_test)}')
  
  # --------------------
  # Data Normalization
  # --------------------
  sc = StandardScaler()
  X_train = sc.fit_transform(X_train)
  X_test = sc.transform(X_test)
  
  # --------------
  # Model 
  # --------------
  # knn = KNeighborsClassifier(n_neighbors=3)
  knn = KNN(k=3, distance_metric='euclidean', type='classification')
  
  # --------------
  # Train
  # --------------
  knn.fit(X_train, y_train)
  y_pred = knn.predict(X_test)
  
  # --------------
  # Evaluation
  # --------------
  cm = confusion_matrix(y_test, y_pred)
  print(f'Accuracy: {accuracy_score(y_test, y_pred):.2%}')
  
  print("Classification Report:")
  print(classification_report(y_test, y_pred))
  
  print("Confusion Matrix:")
  print(cm)
  
  # -------------------
  # Visualize the data
  # -------------------
  plot_data(data=df, cols=['sepal length', 'sepal width'], title="Iris Dataset: Sepal Length vs Sepal Width")
  plot_confusion_matrix(cm, classes=le.classes_, outdir=OUTPUT_DIR)
  
if __name__ == "__main__":
  main()
