import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

from sklearn import datasets
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from ucimlrepo import fetch_ucirepo 

from modules.Plots import plot_regression_results
from modules.KNN import KNN


# Percentual Means Absolute Error
def mean_absolute_percentage_error(y_true, y_pred): 
  y_true, y_pred = np.array(y_true), np.array(y_pred)
  return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

ROOT_DIR = Path(__file__).parent.parent.parent
OUTPUT_DIR = ROOT_DIR / 'outputs-results' / 'knn_regression'

print(f"Current working directory: {ROOT_DIR}")

result_mse = []
result_mae = []
result_r2 = []
result_mape = []

def main():
  # -------------------------------------------
  # Create output directory if it doesn't exist
  # -------------------------------------------
  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
  print(f"Output directory: {OUTPUT_DIR}")
  
  # -------------
  # Load Dataset
  # -------------
  # Dataset info: City cycle fuel consumption dataset
  data = fetch_ucirepo(id=9) 
  X = data.data.features
  y = data.data.targets
  
  # ------------------
  # Original dataset
  # ------------------
  df = pd.DataFrame(X, columns=data.data.feature_names)
  df['target'] = y
  print("Original dataset head:")
  print(df.head())

  # ------------------------------------------
  # Split the data into training and test sets
  # ------------------------------------------
  X = X.drop(columns=['horsepower', 'model_year', 'origin'])  # Drop the first column (ID)
  print("X head after dropping columns:")
  print(X.head())
  
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  
  X_train = np.array(X_train)
  X_test = np.array(X_test)
  y_train = np.array(y_train)
  y_test = np.array(y_test)
  
  print(f'Training set size: {X_train.shape} and {y_train.shape}')
  print(f'Test set size: {X_test.shape} and {y_test.shape}')
  
  # --------------------
  # Data Normalization
  # --------------------
  sc = StandardScaler()
  X_train = sc.fit_transform(X_train)
  X_test = sc.transform(X_test)
  
  
  # --------------
  # Model 
  # --------------
  # knn = KNeighborsRegressor(n_neighbors=3, metric='euclidean')
  knn = KNN(k=3, distance_metric='euclidean', type='regression')
  
  # ------------------------------
  # TRAIN k-fold cross-validation
  # ------------------------------
  for i, (train_idx, test_idx) in enumerate(KFold(n_splits=5, shuffle=True, random_state=42).split(X_train)):
    knn.fit(X_train[train_idx], y_train[train_idx])
    y_pred = knn.predict(X_train[test_idx])
    
    mse = mean_squared_error(y_train[test_idx], y_pred)
    mae = mean_absolute_error(y_train[test_idx], y_pred)
    r2 = r2_score(y_train[test_idx], y_pred)
    mape = mean_absolute_percentage_error(y_train[test_idx], y_pred)

    result_mse.append(mse)
    result_mae.append(mae)
    result_r2.append(r2)
    result_mape.append(mape)
  
  print(f'Mean Squared Error: {np.mean(result_mse):.2f}')
  print(f'Root Mean Squared Error: {np.sqrt(np.mean(result_mse)):.2f}')
  print(f'Mean Absolute Error: {np.mean(result_mae):.2f}')
  print(f'Mean Absolute Percentage Error: {np.mean(result_mape):.2f}%')
  print(f'R^2 Score: {np.mean(result_r2):.2f}')
  
  # ----------------------------
  # Plot results
  # ----------------------------
  
  y_pred = knn.predict(X_test)
  
  plot_regression_results(
    X_train=sc.inverse_transform(X_train), 
    y_train=y_train, 
    X_test=sc.inverse_transform(X_test), 
    y_test=y_test, 
    y_pred=y_pred,
    outdir=OUTPUT_DIR
  )
  
if __name__ == "__main__":
  main()
