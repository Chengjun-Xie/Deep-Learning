import numpy as np
from Layers import Base
from Layers import Helpers
import copy


class BatchNormalization(Base.base):
    def __init__(self, channels):
        super().__init__()

        # channels denotes the number of channels of the input tensor in both, the vector and the image-case
        self.channels = channels
        self.weights = None
        self.bias = None
        self.input_shape = None

        self.alpha = 0.8
        self.preMean = 0
        self.preVariance = 0
        self.updated = False

        self.normed_input_tensor = None
        self._optimizer = None
        self.weights_optimizer = None
        self.bias_optimizer = None

        self.gradient_weights = None
        self.gradient_bias = None
        self.input_tensor = None

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value
        self.weights_optimizer = copy.deepcopy(self._optimizer)
        self.bias_optimizer = copy.deepcopy(self._optimizer)

    def initialize(self, weights_initializer=None, bias_initializer=None):
        self.weights = np.ones(self.channels)
        self.bias = np.zeros(self.channels)

    def forward(self, input_tensor):
        # initialization
        if not self.updated:
            self.initialize()
        self.input_shape = input_tensor.shape

        # for convolutinoal layer
        if len(self.input_shape) is 4:
            input_tensor = self.reformat(input_tensor)
        self.input_tensor = input_tensor

        # test phase
        if self.phase == Base.Phase.test:
            input_tensor -= self.preMean
            input_tensor /= np.sqrt(self.preVariance + np.finfo(float).eps)
            if len(self.input_shape) is 4:
                input_tensor = self.reformat(input_tensor)
            return input_tensor

        # Normalization
        mean = np.mean(input_tensor, axis=0)
        variance = np.var(input_tensor, axis=0)
        eps = np.finfo(float).eps
        output_tensor = input_tensor - mean
        output_tensor /= np.sqrt(variance + eps)
        self.normed_input_tensor = output_tensor

        output_tensor = self.weights * output_tensor + self.bias

        if len(self.input_shape) is 4:
            # reverse
            output_tensor = self.reformat(output_tensor)

        # moving average estimation
        # to calculate the true training set mean and variance
        if not self.updated:
            # Initialize mean and variance with the batch mean and the batch standard deviation
            # of the first batch used for training.
            self.preMean = mean
            self.preVariance = variance
        else:
            self.preMean = self.alpha * self.preMean + (1 - self.alpha) * mean
            self.preVariance = self.alpha * self.preVariance + (1 - self.alpha) * variance
        self.updated = True

        return output_tensor

    def backward(self, error_tensor):
        # error_tensor has the same shape of input_tensor

        # for convolutinoal layer
        if len(self.input_shape) is 4:
            error_tensor = self.reformat(error_tensor)

        self.gradient_weights = (error_tensor * self.normed_input_tensor).sum(axis=0)
        self.gradient_bias = error_tensor.sum(axis=0)
        gradient_input = Helpers.compute_bn_gradients(error_tensor,
                                                      self.input_tensor,
                                                      self.weights,
                                                      self.preMean,
                                                      self.preVariance)

        if len(self.input_shape) is 4:
            # reverse
            gradient_input = self.reformat(gradient_input)

        # update
        if self._optimizer is not None:
            self.weights = self.weights_optimizer.calculate_update(self.weights, self.gradient_weights)
            self.bias = self.bias_optimizer.calculate_update(self.bias, self.gradient_bias)

        return gradient_input

    def reformat(self, tensor):
        """
        the method reformats the image-like tensor (with 4 dimension) into its vector-like variant (with 2 dimensions)
        and the same method reformats the vector-like tensor into its image-like tensor variant.
        """
        if type(tensor) is None:
            # default
            return tensor

        if len(tensor.shape) is 4:
            tensor = tensor.reshape((*tensor.shape[:2], np.prod(tensor.shape[2:])))     # (B, H, M, N) -> (B, H, M*N)
            tensor = np.transpose(tensor, (0, 2, 1))                                    # (B, H, M*N) -> (B, M*N, H)
            tensor = tensor.reshape((np.prod(tensor.shape[:2]), tensor.shape[2]))       # (B, M*N, H) -> (B*M*N, H)
        elif len(tensor.shape) is 2:
            # (B, M*N, H) <- (B*M*N, H)
            tensor = tensor.reshape((self.input_shape[0], np.prod(self.input_shape[2:]), self.input_shape[1]))
            tensor = np.transpose(tensor, (0, 2, 1))                                    # (B, H, M*N) <- (B, M*N, H)
            tensor = tensor.reshape(self.input_shape)                                   # (B, H, M, N) <- (B, H, M*N)
        return tensor









