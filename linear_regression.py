import numpy as np


class LinearRegression:
  def __init__(self, learning_rate=1e-3, epochs=1000, batch_size=1):
    self.learning_rate = learning_rate
    self.epochs = epochs
    self.batch_size = batch_size
    self.theta = None
    self.loss_history = []


  def fit(self, X, y):
    n, p = X.shape
    X_b = np.c_[np.ones((n, 1)), X] # shape: (n, p+1)
    self.theta = np.random.randn(p + 1, 1)

    nb_steps = 0
    for epoch in range(self.epochs):
      # for better performance, shuffle data:
      indices = np.random.permutation(n)
      X_b_shuffled = X_b[indices]
      y_shuffled = y[indices]

      for i in range(0, n, self.batch_size):
        X_batch = X_b_shuffled[i: i+self.batch_size]
        y_batch = y_shuffled[i: i+self.batch_size]

        gradients = (2 / self.batch_size) * X_batch.T @ (X_batch @ self.theta - y_batch)

        self.theta = self.theta - self.learning_rate * gradients

      mse = np.mean((X_b @ self.theta - y)**2)
      self.loss_history.append(mse)
      

  def predict(self, X):
    n = X.shape[0]
    X_b = np.c_[np.ones((n, 1)), X] # add bias term
    return X_b @ self.theta
