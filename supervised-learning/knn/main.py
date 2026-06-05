import pandas as pd
import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from modules.Plots import plot_data, plot_decision_boundary, plot_inference
from modules.KNN import KNN

def main():
  # Load diabetes dataset
  data = datasets.load_iris()
  # diabetes = datasets.load_diabetes()
  # ata = datasets.fetch_covtype()  # Forest CoverType dataset
  X = data.data[:, :2]  # Sepal length & width
  y = data.target
  
  # Use only Setosa and Versicolor
  X = X[y != 2]
  y = y[y != 2]
  
  df = pd.DataFrame(X, columns=data.feature_names[:2])
  df['target'] = y
  
  print(df.head())

  # Split the data into training and test sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  
  # Standardize the features
  # sc = StandardScaler()
  # X_train_std = sc.fit_transform(X_train)
  # X_test_std = sc.transform(X_test)
  
  # Create KNN classifier
  # knn = KNeighborsClassifier(n_neighbors=3)
  knn = KNN(k=3, distance_metric='euclidean')
  
  knn.fit(X_train, y_train)
  y_pred = knn.predict(X_test)
  
  # Visualize the data
  plot_data(data=df, cols=['sepal length (cm)', 'sepal width (cm)'], title="Iris Dataset: Sepal Length vs Sepal Width")
  
  # Visualize the decision boundary
  plot_decision_boundary(X_train, y_train, knn)  
  
  # Print the accuracy of the classifier
  print(f'Accuracy: {accuracy_score(y_test, y_pred):.2%}')
  
if __name__ == "__main__":
  main()
