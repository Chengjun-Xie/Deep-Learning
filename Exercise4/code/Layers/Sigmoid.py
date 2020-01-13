import numpy as np
from Layers import Base


class Sigmoid(Base.base):
    def __init__(self):
        super().__init__()
        self.activation = None

    def forward(self, input_tensor):
        self.activation = 1 / (1 + np.exp(-input_tensor))
        return self.activation

    def backward(self, error_tensor):
        return self.activation * (1 - self.activation) * error_tensor
