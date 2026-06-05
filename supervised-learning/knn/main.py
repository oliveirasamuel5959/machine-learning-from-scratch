import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

from modules.Plots import plot_data
from modules.KNN import KNN

def main():
  # Load the built-in Iris dataset
  iris = load_iris()
  X, y = iris.data, iris.target
  
  print("First 5 samples of the dataset:")
  print(X[:5])
  
  plot_data(X, y)

if __name__ == "__main__":
  main()
