import numpy as np
from Layers import Base


class Dropout(Base.base):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability
        self.mask = None

    def forward(self, input_tensor):
        """
        Dropout: Randomly set activations to zero with probability p
        """
        if self.phase == Base.Phase.test:
            return input_tensor

        self.mask = np.random.random(input_tensor.shape) > self.probability
        input_tensor[self.mask] = 0
        return input_tensor * (1 / self.probability)

    def backward(self, error_tensor):
        error_tensor[self.mask] = 0
        return error_tensor