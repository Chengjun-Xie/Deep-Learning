import numpy as np
import matplotlib.pyplot as plt


class Sgd:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        return weight_tensor - self.learning_rate * gradient_tensor


class SgdWithMomentum:
    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.prevMomentum = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        curMomentum = self.momentum_rate * self.prevMomentum - self.learning_rate * gradient_tensor
        weight_tensor = weight_tensor + curMomentum
        self.prevMomentum = curMomentum
        return weight_tensor


class Adam:
    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho

        self.preV = 0
        self.preR = 0
        self.k = 1

    def calculate_update(self, weight_tensor, gradient_tensor):
        g = gradient_tensor
        v = self.mu * self.preV + (1 - self.mu) * g
        r = self.rho * self.preR + (1 - self.rho) * g * g

        v_hat = v / (1 - self.mu ** self.k)
        r_hat = r / (1 - self.rho ** self.k)

        eps = np.finfo(float)
        weight_tensor -= self.learning_rate * ((v_hat + eps.eps) / (np.sqrt(r_hat) + eps.eps))

        self.preV = v
        self.preR = r
        self.k += 1
        return weight_tensor
