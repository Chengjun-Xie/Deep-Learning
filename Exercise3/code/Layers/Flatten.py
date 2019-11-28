import numpy as np


class Flatten:
    def __init__(self):
        self.input_shape = ()

    """
    flatten a n dimensional array into 2D array     
    param{input_tensor} ∈ batch_size * input_shape(multidimensional array)
    return ∈ batch_size * 1D shape
    """
    def forward(self, input_tensor):
        # np.prod() : Return the product of array elements over a given axis.
        self.input_shape = input_tensor.shape[1:]
        return input_tensor.reshape(input_tensor.shape[0], np.prod(self.input_shape))

    """
    fold the 2D array into n dimensional array
    param{error_tensor} ∈ batch_size * 1D shape
    return ∈ batch_size * input_shape(multidimensional array)
    """
    def backward(self, error_tensor):
        # *-operation : convert tuple into separated values
        return error_tensor.reshape(error_tensor.shape[0], *self.input_shape)
