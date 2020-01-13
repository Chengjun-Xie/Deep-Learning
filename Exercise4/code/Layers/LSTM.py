import numpy as np
from Layers import Base
from Layers import FullyConnected
from Layers import Sigmoid
from Layers import TanH
from Optimization import *

"""
General strategy for RNN:
    We interpret the batch dimension as time dimension now :
    Samples are correlated in this dimension
    This allows to reuse loss functions, optimizers, initializers,
    activation functions and the Neural Network class
"""


class LSTM(Base.base):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size  # dimension of the input vector
        self.hidden_size = hidden_size  # dimension of the hidden state
        self.output_size = output_size

        self.__memorize = False

        self.preHidden_state = None
        self.hidden_state = None

        # neural networks
        network_size = self.input_size + self.hidden_size
        self.cell_layer = FullyConnected.FullyConnected(network_size, self.hidden_size)
        self.update_gate = FullyConnected.FullyConnected(network_size, self.hidden_size)
        self.forget_gate = FullyConnected.FullyConnected(network_size, self.hidden_size)
        self.output_gate = FullyConnected.FullyConnected(network_size, self.hidden_size)
        self.final_layer = FullyConnected.FullyConnected(self.hidden_size, self.output_size)

        # activation function
        self.tanh_cell = TanH.TanH()
        self.sigmoid_update = Sigmoid.Sigmoid()
        self.sigmoid_forget = Sigmoid.Sigmoid()
        self.sigmoid_output = Sigmoid.Sigmoid()
        self.tanh_hidden = TanH.TanH()
        self.sigmoid_final = Sigmoid.Sigmoid()

        self._optimizer = None

        # helper variables for calculations in backward pass
        self.output_gate_tensor = None
        self.cell_to_hidden_tensor = None
        self.update_tensor = None
        self.forget_tensor = None
        self.cell_state = None
        self.cell_tensor = None
        self.gradient_cell = np.zeros(self.hidden_size)


    @property
    def memorize(self):
        return self.__memorize

    @memorize.setter
    def memorize(self, value):
        self.__memorize = value

    @property
    def gradient_weights(self):
        return np.hstack((self.cell_layer.gradient_weights,
                          self.update_gate.gradient_weights,
                          self.forget_gate.gradient_weights,
                          self.output_gate.gradient_weights))

    @property
    def weights(self):
        return np.hstack((self.cell_layer.weights,
                          self.update_gate.weights,
                          self.forget_gate.weights,
                          self.output_gate.weights))

    @weights.setter
    def weights(self, value):
        self.cell_layer.weights = value[:, :self.hidden_size]
        self.update_gate.weights = value[:, self.hidden_size: self.hidden_size*2]
        self.forget_gate.weights = value[:, self.hidden_size*2: self.hidden_size*3]
        self.output_gate.weights = value[:, self.hidden_size*3:]

    def initialize(self, weights_initializer, bias_initializer):
        self.cell_layer.initialize(weights_initializer, bias_initializer)
        self.update_gate.initialize(weights_initializer, bias_initializer)
        self.forget_gate.initialize(weights_initializer, bias_initializer)
        self.output_gate.initialize(weights_initializer, bias_initializer)
        self.final_layer.initialize(weights_initializer, bias_initializer)

    def forward(self, input_tensor):
        if not self.__memorize or self.preHidden_state is None:
            self.hidden_state = np.zeros((input_tensor.shape[0] + 1, self.hidden_size))
        else:
            self.hidden_state[0] = self.preHidden_state

        # initialize
        output_tensor = np.zeros((input_tensor.shape[0], self.output_size))
        preCell_state = np.zeros(self.hidden_size)

        self.output_gate_tensor = np.zeros((input_tensor.shape[0], self.hidden_size))
        self.cell_to_hidden_tensor = np.zeros((input_tensor.shape[0], self.hidden_size))
        self.update_tensor = np.zeros((input_tensor.shape[0], self.hidden_size))
        self.forget_tensor = np.zeros((input_tensor.shape[0], self.hidden_size))
        self.cell_state = np.zeros((input_tensor.shape[0], self.hidden_size))
        self.cell_tensor = np.zeros((input_tensor.shape[0], self.hidden_size))

        for t in range(input_tensor.shape[0]):
            self.cell_state[t] = self.preHidden_state

            # Concatenate hidden_state and input ---> [ht−1,xt]
            concatenated_tensor = np.hstack((self.hidden_state[t], input_tensor[t]))

            # temp cell state: ˜Ct = tanh(WC ·[ht−1,xt] + bC)
            cell_tensor = self.cell_layer.forward(concatenated_tensor)
            cell_tensor = self.tanh_cell.forward(cell_tensor)
            self.cell_tensor[t] = cell_tensor

            # update_gate: it = σ (Wi ·[ht−1,xt] + bi)
            update_tensor = self.update_gate.forward(concatenated_tensor)
            update_tensor = self.sigmoid_update.forward(update_tensor)
            self.update_tensor[t] = update_tensor

            # forget_gate: ft = σ (Wf ·[ht−1,xt] + bf)
            forget_tensor = self.forget_gate.forward(concatenated_tensor)
            forget_tensor = self.sigmoid_forget.forward(forget_tensor)
            self.forget_tensor[t] = forget_tensor

            # output_gate:
            output_gate_tensor = self.output_gate.forward(concatenated_tensor)
            output_gate_tensor = self.sigmoid_output.forward(output_gate_tensor)
            self.output_gate_tensor[t] = output_gate_tensor

            # next_hidden_state
            preCell_state = update_tensor * cell_tensor + forget_tensor * preCell_state
            cell_to_hidden_tensor = self.tanh_hidden.forward(preCell_state)
            self.cell_to_hidden_tensor[t] = cell_to_hidden_tensor

            self.hidden_state[t + 1] = output_gate_tensor * cell_to_hidden_tensor
            self.preHidden_state = self.hidden_state[t + 1]

            # output
            output_tensor[t] = self.final_layer.forward(self.hidden_state[t + 1])
            output_tensor[t] = self.sigmoid_final.forward(output_tensor[t])

        return output_tensor

    def backward(self, error_tensor):
        """
        Resource which explained the math behind an LSTM,
        especially the back propagation!!!
        http://arunmallya.github.io/writeups/nn/lstm/index.html#/8
        """
        gradient_hidden = np.zeros(self.hidden_size)
        output = np.zeros((error_tensor.shape[0], self.input_size))
        preHidden = np.zeros(self.hidden_size)
        preCell_state = np.zeros(self.hidden_size)

        for t in range(error_tensor.shape[0]-1, -1, -1):
            # == compute gradient of output_gate and cell state ==
            # Forward pass: hidden<t> = output_gate<t> * tanh(cell<t>)
            # gradient w.r.t to output
            gradient_final = self.sigmoid_final.backward(error_tensor[t])
            gradient_final = self.final_layer.backward(gradient_final)
            gradient_hidden = gradient_final + preHidden

            # gradient w.r.t. output gate
            gradient_output_gate = gradient_hidden * self.cell_to_hidden_tensor[t]
            gradient_output_gate = self.sigmoid_output.backward(gradient_output_gate)
            gradient_output_gate = self.output_gate.backward(gradient_output_gate)

            # gradient w.r.t. next hidden state
            gradient_cell_error = gradient_hidden * self.output_gate_tensor[t]
            gradient_cell_error = self.tanh_hidden.backward(gradient_cell_error)
            self.gradient_cell = gradient_cell_error + preCell_state

            # == LSTM memory cell update ==
            # Forward pass: cell<t> = update_gate<t> * ~cell<t> + forget_gate<t> * cell<t-1>
            # gradient w.r.t. update gate
            gradient_update = self.gradient_cell * self.cell_tensor[t]
            gradient_update = self.sigmoid_update.backward(gradient_update)
            gradient_update = self.update_gate.backward(gradient_update)

            # gradient w.r.t. temp cell state
            gradient_cell_tensor = self.gradient_cell * self.update_tensor[t]
            gradient_cell_tensor = self.tanh_hidden.backward(gradient_cell_tensor)
            gradient_cell_tensor = self.cell_layer.backward(gradient_cell_tensor)

            # gradient w.r.t. forget gate
            gradient_forget = self.gradient_cell * self.cell_state[t]
            gradient_forget = self.sigmoid_forget.backward(gradient_forget)
            gradient_forget = self.forget_gate.backward(gradient_forget)

            # gradient w.r.t. previous cell state
            gradient_preCell = self.gradient_cell * self.forget_tensor[t]
            preCell_state = gradient_preCell

            # == input and previous hidden state
            gradient_input = gradient_forget + gradient_update + gradient_output_gate + gradient_cell_tensor
            output[t] = gradient_input[0, self.hidden_size:]
            preHidden = gradient_input[0, :self.hidden_size]

        return output









