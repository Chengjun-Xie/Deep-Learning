import numpy as np
import scipy.signal
import copy


class Conv:
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        self.stride_shape = stride_shape    # can be a single value or a tuple

        # determines whether this objects provides a 1D or a 2D convolution layer.
        # For 1D, it has the shape [c, m]
        # whereas for 2D, it has the shape [c, m, n]
        # where c represents the number of input channels, and m, n represent the spacial extent of the filter kernel.
        self.convolution_shape = convolution_shape

        self.num_kernels = num_kernels  # integer value

        self.weights = np.random.rand(self.num_kernels, *self.convolution_shape)
        self.bias = np.random.rand(self.num_kernels)    # element-wise addition of a scalar value for every kernel

        self.gradient_weights = np.zeros(self.weights.shape)
        self.gradient_bias = np.zeros(self.bias.shape)
        self.input_tensor = None

        self._optimizer = None  # optimizations method

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    """
    return input tensor for the next layer. 
    param{input_tensor} The input layout for 1D is defined in b, c, y order, for 2D in b, c, y, x order.
                        b stands for the batch, c represents the channels and x, y represent the spatial dimensions.
    return  Use zero-padding for convolutions/correlations (“same” padding). 
            This allows input and output to have the same spacial shape for a stride of 1. 
    """
    def forward(self, input_tensor):
        self.input_tensor = input_tensor

        batch = []
        for i in range(input_tensor.shape[0]):
            channel = []
            for j in range(self.num_kernels):

                # output of this methods is the same size as input_tensor[i], which is (c, y, x) or (c, y)
                # it will do "zero-padding" no only along the x-y plane, but also along the channel
                # only the middle channel is valid
                tensor = scipy.signal.correlate(input_tensor[i], self.weights[j], 'same')

                valid_channel = self.convolution_shape[0] // 2
                tensor = tensor[valid_channel]
                tensor += self.bias[j]

                # stride (upsampling)
                if len(self.stride_shape) == 1:
                    tensor = tensor[::self.stride_shape[0]]
                else:
                    tensor = tensor[::self.stride_shape[0], ::self.stride_shape[1]]

                channel.append(tensor)
            batch.append(channel)
        output = np.array(batch)

        return output

    """
    updates the parameters and returns the error tensor for the next layer
    param{error_tensor} ∈ (b, num_kernel, spatial dimension of error_tensor)
    return ∈ (b, channel, spatial dimension of input_tensor)
    """
    def backward(self, error_tensor):
        # initialize gradient_weights and gradient_bias !!!
        self.gradient_weights = np.zeros(self.weights.shape)
        self.gradient_bias = np.zeros(self.bias.shape)

        # ==== Part1: Gradient with respect to lower layers ====
        """
        kernel modify
        input:  weights ∈ (num_kernel, channel, spatial dimension)
        output: kernel ∈ (channel, num_kernel, spatial dimension)
        """
        weights_channel = []
        for i in range(self.convolution_shape[0]):
            weights_kernel = []
            for j in range(self.num_kernels):
                weights_kernel.append(self.weights[j, i])
            weights_channel.append(weights_kernel)
        gradient_layer_kernel = np.array(weights_channel)
        gradient_layer_kernel = np.flip(gradient_layer_kernel, axis=1)  # channel dimension needs to be flipped

        # compute gradients
        batch = []
        for i in range(error_tensor.shape[0]):
            channel = []

            """
            re-stride
            input：  error_tensor ∈ (b, n, spatial dimension of error_tensor)
            output： ∈ (b, n, spatial dimension of input_tensor)
            """
            error_tensor_restride = np.zeros((error_tensor.shape[1], *self.input_tensor.shape[2:]))
            if len(self.stride_shape) == 1:
                error_tensor_restride[:, ::self.stride_shape[0]] = error_tensor[i]
            else:
                error_tensor_restride[:, ::self.stride_shape[0], ::self.stride_shape[1]] = error_tensor[i]

            # convolution along input_channel
            for j in range(gradient_layer_kernel.shape[0]):
                tensor = scipy.signal.convolve(error_tensor_restride, gradient_layer_kernel[j], 'same')
                valid_channel = error_tensor.shape[1] // 2
                tensor = tensor[valid_channel]
                channel.append(tensor)

            # ==== Part2:  Gradient with respect to the weights
            # convolution along num_kernel
            gradient_kernel = []
            gradient_bias = []

            # padding
            if len(self.stride_shape) == 1:
                padding_shape = ((0, 0),
                                 (self.convolution_shape[1] // 2, (self.convolution_shape[1] - 1) // 2))
            else:
                padding_shape = ((0, 0),
                                 (self.convolution_shape[1] // 2, (self.convolution_shape[1] - 1) // 2),
                                 (self.convolution_shape[2] // 2, (self.convolution_shape[2] - 1) // 2))
            input_tensor_padding = np.pad(self.input_tensor[i], padding_shape, 'constant')

            for k in range(self.num_kernels):
                # weights
                # expand (y, x) into (1, y, x) to have same dimension as input_tensor_padding
                temp_tensor = np.expand_dims(error_tensor_restride[k], axis=0)
                gradient_tensor = scipy.signal.correlate(input_tensor_padding, temp_tensor, 'valid')
                gradient_kernel.append(gradient_tensor)

                # bias
                gradient_bias.append(error_tensor[i, k].sum())

            gradient_weights = np.array(gradient_kernel)
            gradient_bias = np.array(gradient_bias)

            # sum up gradient_weights and gradient_bias of each batch
            self.gradient_weights += gradient_weights
            self.gradient_bias += gradient_bias

            batch.append(channel)
        output_tensor = np.array(batch)

        # update
        if (self._optimizer != None):
            weights_optimizer = copy.deepcopy(self._optimizer)
            bias_optimizer = copy.deepcopy(self._optimizer)
            self.weights = weights_optimizer.calculate_update(self.weights, self.gradient_weights)
            self.bias = bias_optimizer.calculate_update(self.bias, self.gradient_bias)

        return output_tensor

    """
     reinitializing its weights and bias

     For convolutional layers: 
     “fan_in”: [# input channels × kernel height × kernel width] 
     “fan_out”: [# output channels × kernel height x kernel width]

     @param{weights_initializer} pass in a object of Initializer
     @param{bias_initializer} pass in a object of Initializer
    """
    def initialize(self, weights_initializer, bias_initializer):
        weights_shape = self.weights.shape
        weights_fan_in = np.prod(self.convolution_shape)
        weights_fan_out = self.num_kernels * np.prod(self.convolution_shape[1:])
        self.weights = weights_initializer.initialize(weights_shape, weights_fan_in, weights_fan_out)

        bias_shape = self.bias.shape
        bias_fan_in = 1
        bias_fan_out = self.num_kernels
        self.bias = bias_initializer.initialize(bias_shape, bias_fan_in, bias_fan_out)
