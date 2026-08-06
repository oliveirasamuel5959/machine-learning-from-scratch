import numpy as np
from mlscratch.supervised_learning.perceptron.modules.Perceptron import Perceptron

def main():
  X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
  y = np.array([0, 1, 1, 1])  # OR gate

  perceptron = Perceptron(learning_rate=0.1, epochs=1000)
  perceptron.fit(X, y)

  test_cases = [[0, 0], [0, 1], [1, 0], [1, 1]]
  
  print(f"\nFinal Weights and Bias: {perceptron.get_weights()}")
  
  print("\n--- Predictions ---")
  for test in test_cases:
    pred, prob = perceptron.predict(np.array(test))
    print(f"Input: {test} | Predicted: {pred[0]} | Probability: {prob[0]:.4f}")

if __name__ == "__main__":
    main()