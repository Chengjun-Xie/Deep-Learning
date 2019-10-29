import numpy as np
import matplotlib.pyplot as plt


class Sgd:
    def __init__(self, learning_rate=1e-9):
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        return weight_tensor - self.learning_rate * gradient_tensor
