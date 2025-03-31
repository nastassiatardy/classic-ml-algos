import numpy as np


class LogisticRegression:
  def __init__(self, learning_rate=1e-3, epochs = 1000):
    self.learning_rate = learning_rate
    self.epochs = epochs
    self.theta = None
    self.loss_history = []


  def sigmoid(self, z):
    return 1 / (1 + np.exp(-z))


  def cost_function(self, X, y):
    n = len(y)
    predictions = self.sigmoid(X @ self.theta)

    cost = - (1/n) * (y.T @ np.log(predictions) + (1 - y).T @ np.log(1 - predictions))
    return cost[0][0]


  def fit(self, X, y):
    n, p = X.shape
    X_b = np.c_[np.ones((n, 1)), X]

    self.theta = np.random.randn(p+1, 1)

    for epoch in range(self.epochs):
      predictions = self.sigmoid(X_b @ self.theta)
      gradients = (1/n) * X_b.T @ (predictions - y)
      self.theta -= self.learning_rate * gradients

      cost = self.cost_function(X_b, y)
      self.loss_history.append(cost)


  def predict(self, X):
    n = X.shape[0]
    X_b = np.c_[np.ones((n, 1)), X] # add bias term
    probabilities = self.sigmoid(X_b @ self.theta)
    return (probabilities >= 0.5).astype(int) # convert to binary prediction
    