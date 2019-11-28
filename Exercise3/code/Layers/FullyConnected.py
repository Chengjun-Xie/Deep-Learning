import numpy as np
from Optimization import Optimizers


class FullyConnected:
    def __init__(self, input_size, output_size):
        self.input_size = input_size    # n
        self.output_size = output_size  # m
        self._optimizer = None  # optimizations method

        # The member for the weights and biases should be named weights
        # weights ∈ (n+1)*m
        self.weights = np.random.rand(self.input_size + 1, self.output_size )

        # returns the gradient with respect to the weights, after they have been calculated in the backward-pass
        self.gradient_weights = np.zeros((self.output_size, self.input_size))

    def get_gradient_weights(self):
        return self.gradient_weights

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    """
     reinitializing its weights and bias
     
     For fully connected layers: 
     “fan_in”: input dimension of the weights 
     “fan_out”: output dimension of the weights
    
     @param{weights_initializer} pass in a object of Initializer
     @param{bias_initializer} pass in a object of Initializer
    """
    def initialize(self, weights_initializer, bias_initializer):
        weights_shape = (self.input_size, self.output_size)
        weights_fan_in = self.input_size
        weights_fan_out = self.output_size
        weights = weights_initializer.initialize(weights_shape, weights_fan_in, weights_fan_out)

        bias_shape = (1, self.output_size)
        bias_fan_in = 1
        bias_fan_out = self.output_size
        bias = bias_initializer.initialize(bias_shape, bias_fan_in, bias_fan_out)

        self.weights = np.vstack((weights, bias))

    """
    returns the input tensor for the next layer.
    
    @param{input_tensor} ∈ batch_size * input_size
    @return ∈  batch_size * output_size
    """
    def forward(self, input_tensor):
        # Add a bias unit to x∈RN by adding a dimension with xn+1 = 1
        # memory layout: X.T * W.T = Y.T
        x = np.ones((input_tensor.shape[0], 1))
        x = np.hstack((input_tensor, x))
        z = x.dot(self.weights)

        # store x to gradient_weight for backward propagation
        self.gradient_weights = np.copy(x.T)
        return z

    """
    returns the error tensor for the next layer
    Use the method calculate_update(weight tensor, gradient tensor) of your optimizer in your backward pass,
    in order to update your weights.

    @param{error_tensor} ∈ batch_size * output_size
    @return ∈ batch_size * input_size
    """
    def backward(self, error_tensor):
        # memory layout: w.T(t+1) = w.T(t) - eta * X * En.T
        self.gradient_weights = self.gradient_weights.dot(error_tensor)

        # weights_backward ∈ n*m
        # memory layout: En-1.T = En.T * W
        weights_backward = self.weights[:self.input_size, :]
        error_tensor = error_tensor.dot(weights_backward.T)

        if(self._optimizer != None):
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights)

        return error_tensor

