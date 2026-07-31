import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

from sklearn.base import clone
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from ucimlrepo import fetch_ucirepo 

from modules.Plots import plot_regression_line, plot_regression_results_all_features
from modules.utils import save_metrics
from modules.KNN import KNN


# Percentual Means Absolute Error
def mean_absolute_percentage_error(y_true, y_pred): 
  y_true, y_pred = np.array(y_true), np.array(y_pred)
  return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

ROOT_DIR = Path(__file__).parent.parent.parent
OUTPUT_DIR = ROOT_DIR / 'outputs-results' / 'knn-regression'

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
  print("\nOriginal dataset head:")
  print(df.head())

  # ------------------------------------------
  # Split the data into training and test sets
  # ------------------------------------------
  # Drop origin column
  X = df.drop(columns=['origin'])
  print("\nX head after dropping 'origin' column:")
  print(X.head())
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  
  X_train = np.array(X_train)
  X_test = np.array(X_test)
  y_train = np.array(y_train)
  y_test = np.array(y_test)
  
  print(f'\nTraining set size: {X_train.shape} and {y_train.shape}')
  print(f'Test set size: {X_test.shape} and {y_test.shape}')
  
  # --------------
  # Build Model 
  # --------------
  # knn = KNeighborsRegressor(n_neighbors=3, metric='euclidean')
  knn = KNN(k=3, distance_metric='euclidean', type='regression')
  
  # ------------------------------
  # TRAIN k-fold cross-validation
  # ------------------------------
  kfold = KFold(n_splits=5, shuffle=True, random_state=42)
  
  for i, (train_idx, test_idx) in enumerate(kfold.split(X_train)):
    # Create a fresh clone of the KNN model for each fold to ensure independence
    clone_knn = clone(knn)
    
    # Data Normalization
    scaler = StandardScaler()
    
    # Create fold-specific training and test sets
    X_fold_train = scaler.fit_transform(X_train[train_idx])
    y_fold_train = y_train[train_idx]
    X_fold_test = scaler.transform(X_train[test_idx])
    y_fold_test = y_train[test_idx]

    # Fit the model and predict
    clone_knn.fit(X_fold_train, y_fold_train)
    y_pred = clone_knn.predict(X_fold_test)

    # Calculate metrics
    mse = mean_squared_error(y_fold_test, y_pred)
    mae = mean_absolute_error(y_fold_test, y_pred)
    r2 = r2_score(y_fold_test, y_pred)
    mape = mean_absolute_percentage_error(y_fold_test, y_pred)
    
    print(f"\nFold {i+1} - MSE: {mse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}, MAPE: {mape:.2f}%")

    result_mse.append(mse)
    result_mae.append(mae)
    result_r2.append(r2)
    result_mape.append(mape)
  
  print("\n── Cross-Validation Results ──")
  print(f"MSE : {np.mean(result_mse):.2f} ± {np.std(result_mse):.2f}")
  print(f'RMSE: {np.sqrt(np.mean(result_mse)):.2f}')
  print(f"MAE : {np.mean(result_mae):.2f} ± {np.std(result_mae):.2f}")
  print(f"R²  : {np.mean(result_r2):.2f} ± {np.std(result_r2):.2f}")
  print(f"MAPE: {np.mean(result_mape):.2f}% ± {np.std(result_mape):.2f}%")
  
  save_metrics(
    mse=np.mean(result_mse), 
    mae=np.mean(result_mae), 
    r2=np.mean(result_r2), 
    mape=np.mean(result_mape), 
    outdir=OUTPUT_DIR
  )
  
  # ----------------------------
  # Plot results
  # ----------------------------
  final_scaler = StandardScaler()
  X_train_scaled = final_scaler.fit_transform(X_train)
  X_test_scaled  = final_scaler.transform(X_test)

  knn.fit(X_train_scaled, y_train)
  y_pred = knn.predict(X_test_scaled)
  
  plot_regression_results_all_features(
    X_train=X_train, 
    y_train=y_train, 
    X_test=X_test, 
    y_test=y_test, 
    y_pred=y_pred,
    outdir=OUTPUT_DIR
  )
  
  plot_regression_line(
    y_test=y_test, 
    y_pred=y_pred, 
    outdir=OUTPUT_DIR
  )
  
if __name__ == "__main__":
  main()
