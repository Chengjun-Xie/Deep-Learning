import numpy as np


class Pooling:
    def __init__(self, stride_shape, pooling_shape):
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.input_shape = None
        self.location = None

    """
    returns the input tensor for the next layer.
    param{input_tensor} ∈ (batch, channel, y, x)
    return ∈ (batch, channel, (1 + (y - pooling) // stride), (1 + (x - pooling) // stride))
    """
    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape

        # reshape output height and width
        W = 1 + (input_tensor.shape[2] - self.pooling_shape[0]) // self.stride_shape[0]
        L = 1 + (input_tensor.shape[3] - self.pooling_shape[1]) // self.stride_shape[1]
        output_shape = (*input_tensor.shape[:2], W, L)

        self.location = np.zeros((*input_tensor.shape, 4))
        output_tensor = np.zeros(output_shape)

        for i in range(input_tensor.shape[0]):
            for j in range(input_tensor.shape[1]):
                y_index = 0

                for n in range(0, input_tensor.shape[2], self.stride_shape[0]):
                    x_index = 0
                    for m in range(0, input_tensor.shape[3], self.stride_shape[1]):

                        # only store the result when the pooling box is in of the boundary
                        if (n + self.pooling_shape[0] <= input_tensor.shape[2]) \
                                and (m + self.pooling_shape[1] <= input_tensor.shape[3]):

                            # pooling kernel
                            kernel = input_tensor[i, j, n: n + self.pooling_shape[0], m: m + self.pooling_shape[1]]
                            output_tensor[i, j, y_index, x_index] = np.max(kernel)

                            # this method return the location of the local maximum in pooling kernel
                            location = np.unravel_index(np.argmax(kernel), kernel.shape)
                            self.location[i, j, y_index, x_index, 0] = i
                            self.location[i, j, y_index, x_index, 1] = j
                            self.location[i, j, y_index, x_index, 2] = n + location[0]
                            self.location[i, j, y_index, x_index, 3] = m + location[1]

                        x_index += 1
                    y_index += 1
        return output_tensor

    """
    returns the error tensor for the next layer.
    param{error_tensor} ∈ (batch, channel, (1 + (y - pooling) // stride), (1 + (x - pooling) // stride))
    return ∈ (batch, channel, y, x)
    """
    def backward(self, error_tensor):
        output_tensor = np.zeros(self.input_shape)

        for i in range(error_tensor.shape[0]):
            for j in range(error_tensor.shape[1]):
                for n in range(error_tensor.shape[2]):
                    for m in range(error_tensor.shape[3]):
                        # map a double list into integer tuple
                        location = self.location[i, j, n, m]
                        location = tuple(map(int, location))
                        output_tensor[location] += error_tensor[i, j, n, m]

        return output_tensor
