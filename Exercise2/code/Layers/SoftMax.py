import numpy as np
from Optimization import Loss


class SoftMax:
    def __init__(self):
        self.prediction = None

    """
    returns the estimated class probabilities for each row representing an element of the batch. 

    @param{input_tensor} ∈ batch_size * categories 
    return ∈ batch_size * categories 
    """
    def forward(self, input_tensor):
        input_tensor -= np.max(input_tensor)
        exp = np.exp(input_tensor)

        # summation every element of the batch
        # Notice: The matrices on both sides of element-wise operator must have the same shape !!
        exp_sum = np.sum(exp, axis=1)
        exp_sum = np.tile(exp_sum, (input_tensor.shape[1], 1))

        self.prediction = exp / exp_sum.T
        return self.prediction

    """
    returns the error tensor for the next layer  
    En−1 = ^y * (En − np.sum(En * ^y))
    
    @param{label_tensor} ∈ batch_size * categories 
    return ∈ batch_size * categories 
    """
    def backward(self, label_tensor):
        # summation every element of the batch
        result = label_tensor * self.prediction
        result = np.sum(result, axis=1)
        result = np.tile(result, (label_tensor.shape[1], 1))

        # ^y * (En − sum)
        result = label_tensor - result.T
        result = self.prediction * result
        return result


