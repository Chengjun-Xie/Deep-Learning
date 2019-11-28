import numpy as np

"""
For fully connected layers: 
“fan_in”: input dimension of the weights 
“fan_out”: output dimension of the weights

For convolutional layers: 
“fan_in”: [# input channels × kernel height × kernel width] 
“fan_out”: [# output channels × kernel height x kernel width]
"""

# Default to 0.1
class Constant:
    def __init__(self, const=0.1):
        self.const = const

    def initialize(self, weights_shape, fan_in, fan_out):
        return self.const * np.ones(weights_shape)


# Usually in the range [0,1]
class UniformRandom:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.rand(*weights_shape)


# Zero-mean Gaussian: N(0,σ)
class Xavier:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2 / (fan_out + fan_in))
        return np.random.normal(loc=0, scale=sigma, size=weights_shape)


# zero-mean Gaussian: N(0,σ)
class He:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2 / fan_in)
        return np.random.normal(loc=0, scale=sigma, size=weights_shape)
