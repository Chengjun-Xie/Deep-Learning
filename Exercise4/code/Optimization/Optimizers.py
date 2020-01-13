import numpy as np
import copy


class base:
    """
     Make all optimizers inherit from this ”base-optimizer”.
    """

    def __init__(self):
        self.regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = copy.deepcopy(regularizer)


class Sgd(base):
    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        if (self.regularizer != None):
            output_tensor = weight_tensor - self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
            output_tensor -= self.learning_rate * gradient_tensor
        else:
            output_tensor = weight_tensor - self.learning_rate * gradient_tensor

        return output_tensor


class SgdWithMomentum(base):
    def __init__(self, learning_rate, momentum_rate=0.9):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.prevMomentum = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        curMomentum = self.momentum_rate * self.prevMomentum - self.learning_rate * gradient_tensor

        if (self.regularizer != None):
            weight_tensor -= self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)

        weight_tensor = weight_tensor + curMomentum
        self.prevMomentum = curMomentum
        return weight_tensor


class Adam(base):
    def __init__(self, learning_rate, mu=0.9, rho=0.9):
        super().__init__()
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

        if (self.regularizer != None):
            weight_tensor -= self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
        weight_tensor -= self.learning_rate * ((v_hat + eps.eps) / (np.sqrt(r_hat) + eps.eps))

        self.preV = v
        self.preR = r
        self.k += 1
        return weight_tensor
