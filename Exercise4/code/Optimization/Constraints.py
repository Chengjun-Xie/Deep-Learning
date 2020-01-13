import numpy as np


class L2_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha  # representing the regularization weight

    def calculate_gradient(self, weights):
        return self.alpha * weights

    def norm(self, weights):
        weights = weights.flatten()
        return self.alpha * np.sqrt(weights @ weights.T)


class L1_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha  # representing the regularization weight

    def calculate_gradient(self, weights):
        return self.alpha * np.sign(weights)

    def norm(self, weights):
        return self.alpha * np.absolute(weights).sum()
