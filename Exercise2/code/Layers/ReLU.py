import numpy as np


class ReLU:
    def __init__(self):
        self.input_mask = None

    def forward(self, input_tensor):
        # need to store the input_mask for backward propagation
        self.input_mask = np.ones(input_tensor.shape)
        self.input_mask[input_tensor <= 0] = 0
        return input_tensor * self.input_mask

    """
    En-1 = En  if x > 0
    En-1 = 0   if x <= 0
    """
    def backward(self, error_tensor):
        return error_tensor * self.input_mask
