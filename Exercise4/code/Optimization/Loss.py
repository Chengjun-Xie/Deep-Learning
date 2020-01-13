import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self.input_tensor = None

    """
    returns the loss accumulated over the batch

    @param{input_tensor} ∈ batch_size * categories
    @param{label_tensor} ∈ batch_size * categories
    return loss
    """
    def forward(self, input_tensor, label_tensor):
        self.input_tensor = input_tensor
        eps = np.finfo(float)
        input_tensor = np.log(input_tensor + eps.eps)
        input_tensor[label_tensor == 0] = 0
        loss = - input_tensor.sum()

        return loss

    """
    En = - ^y / y
    """
    def backward(self, label_tensor):
        return -(label_tensor / self.input_tensor)

