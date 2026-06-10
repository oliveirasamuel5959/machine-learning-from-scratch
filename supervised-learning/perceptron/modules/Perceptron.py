import numpy as np

class Perceptron:
  def __init__(self, learning_rate=0.01, epochs=1000):
    self.learning_rate = learning_rate
    self.epochs = epochs
    self.weights = None
    self.bias = None
    
  def _activation(self, x):
    return 1 / (1 + np.exp(-x))
  
  def _binary_cross_entropy(self, y, y_predicted):
    # L=−1/n ∑[ylog(y^​)+(1−y)log(1−y^​)]
    
    # Clip to avoid log(0)
    eps = 1e-15
    y_predicted = np.clip(y_predicted, eps, 1 - eps)
    return -np.mean(y * np.log(y_predicted) + (1 - y) * np.log(1 - y_predicted))
  
  def fit(self, X, y):
    n_samples, n_features = X.shape
    self.weights = np.zeros(n_features)
    self.bias = 0
    
    for epoch in range(self.epochs):
      # Forward pass
      linear_output = np.dot(X, self.weights) + self.bias
      y_predicted = self._activation(linear_output)
      
      # Calculate the error
      error = y - y_predicted
      
      '''
      loss change w.r.t weights and bias using BCE loss
      
      loss = -1/n * sum(y*log(y^) + (1-y)*log(1-y^))
      y^ = activation(z) where z = wX + b
      
      dloss/dw = dloss/dy^ * dy^/dz * dz/dw
      dloss/db = dloss/dy^ * dy^/dz * dz/db
      
      dy^/dz = y^ * (1 - y^)  # derivative of sigmoid
      dz/dw = X
      dz/db = 1
      
      dloss/dw = -1/n * X^T * (y - y_predicted)
      dloss/db = -1/n * sum(y - y_predicted)
      '''
      
      dloss_dw = -np.dot(X.T, error) / n_samples
      dloss_db = -np.sum(error) / n_samples
      
      # Update weights and bias
      self.weights += self.learning_rate * dloss_dw
      self.bias += self.learning_rate * dloss_db
      
      # Log with the correct loss function
      if (epoch + 1) % 100 == 0:
          loss = self._binary_cross_entropy(y, y_predicted)
          print(f'Epoch {epoch+1}/{self.epochs} | BCE Loss: {loss:.6f}')
      
  def predict(self, X):
    # Ensure X is 2D for matrix operations
    X = np.atleast_2d(X)
    
    # Calculate the linear output
    linear_output = np.dot(X, self.weights.T) + self.bias
    
    # Apply the activation function to get the predicted probabilities
    y_predicted = self._activation(linear_output)
    
    y_class = (y_predicted >= 0.5).astype(int)  # shape always matches input
    return y_class, y_predicted
  
  def get_weights(self):
    return self.weights, self.bias
  

      
      