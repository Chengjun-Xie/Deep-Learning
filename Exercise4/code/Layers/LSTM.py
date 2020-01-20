import numpy as np
from Layers import Base
from Layers import FullyConnected
from Layers import Sigmoid
from Layers import TanH

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

        self.hidden_state = None
        self.cell_state = None
        self.preHidden_state_forward = None
        self.preHidden_state_backward = None
        self.preCell_state_forward = np.zeros(self.hidden_size)
        self.preCell_state_backward = np.zeros(self.hidden_size)
        self.first_update = True

        # neural networks
        network_size = self.input_size + self.hidden_size
        self.first_layer = FullyConnected.FullyConnected(network_size, self.hidden_size * 4)
        self.final_layer = FullyConnected.FullyConnected(self.hidden_size, self.output_size)
        self.first_X = []
        self.final_X = []

        # list of activation functions
        self.tanh_cell = []
        self.sigmoid_update = []
        self.sigmoid_forget = []
        self.sigmoid_output = []
        self.tanh_hidden = []
        self.sigmoid_final = []

        self._optimizer = None

        # helper variables for calculations in backward pass
        self.output_gate_tensor = None
        self.cell_to_hidden_tensor = None
        self.update_tensor = None
        self.forget_tensor = None

        self.cell_tensor = None


    @property
    def memorize(self):
        return self.__memorize

    @memorize.setter
    def memorize(self, value):
        self.__memorize = value

    @property
    def gradient_weights(self):
        return self.first_layer.gradient_weights

    @property
    def weights(self):
        return self.first_layer.weights

    @weights.setter
    def weights(self, value):
        self.first_layer.weights = value

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    def initialize(self, weights_initializer, bias_initializer):
        self.first_layer.initialize(weights_initializer, bias_initializer)
        self.final_layer.initialize(weights_initializer, bias_initializer)

    def forward(self, input_tensor):
        if self.first_update:
            # initialize cell state tensor
            self.cell_state = np.zeros((input_tensor.shape[0] + 1, self.hidden_size))
            self.first_update = False

            for t in range(input_tensor.shape[0]):
                # initialize a list of activation function
                self.tanh_cell.append(TanH.TanH())
                self.sigmoid_update.append(Sigmoid.Sigmoid())
                self.sigmoid_forget.append(Sigmoid.Sigmoid())
                self.sigmoid_output.append(Sigmoid.Sigmoid())
                self.tanh_hidden.append(TanH.TanH())
                self.sigmoid_final.append(Sigmoid.Sigmoid())

                # initialize a list of fully connected layers
                self.first_X.append(np.array(0))
                self.final_X.append(np.array(0))

        if not self.__memorize or self.preHidden_state_forward is None:
            # switch between TBPTT and BPTT
            self.hidden_state = np.zeros((input_tensor.shape[0] + 1, self.hidden_size))
            self.cell_state = np.zeros((input_tensor.shape[0] + 1, self.hidden_size))
        else:
            self.hidden_state[0] = self.preHidden_state_forward
            self.cell_state[0] = self.preCell_state_forward

        # initialize
        output_tensor = np.zeros((input_tensor.shape[0], self.output_size))

        self.output_gate_tensor = np.zeros((input_tensor.shape[0], self.hidden_size))
        self.cell_to_hidden_tensor = np.zeros((input_tensor.shape[0], self.hidden_size))
        self.update_tensor = np.zeros((input_tensor.shape[0], self.hidden_size))
        self.forget_tensor = np.zeros((input_tensor.shape[0], self.hidden_size))
        self.cell_tensor = np.zeros((input_tensor.shape[0], self.hidden_size))

        for t in range(input_tensor.shape[0]):
            # for the Notation:
            # self.hidden_state[t] ---> hidden_state at time t-1
            # self.cell_state[t] ---> cell_state at time t-1
            # otherwise: a[t] ----> a at time t

            # Concatenate hidden_state and input ---> [ht−1,xt]
            concatenated_tensor = np.hstack((self.hidden_state[t], input_tensor[t]))
            four_tensors = self.first_layer.forward(concatenated_tensor)
            self.first_X[t] = self.first_layer.X    # store the gradient w.r.t. weights

            # spilt the output of the first FC layer
            forget_tensor = four_tensors[:self.hidden_size]
            update_tensor = four_tensors[self.hidden_size: self.hidden_size * 2]
            cell_tensor = four_tensors[self.hidden_size * 2: self.hidden_size * 3]
            output_gate_tensor = four_tensors[self.hidden_size * 3:]

            # temp cell state: ˜Ct = tanh(WC ·[ht−1,xt] + bC)
            cell_tensor = self.tanh_cell[t].forward(cell_tensor)
            self.cell_tensor[t] = cell_tensor

            # update_gate: it = σ (Wi ·[ht−1,xt] + bi)
            update_tensor = self.sigmoid_update[t].forward(update_tensor)
            self.update_tensor[t] = update_tensor

            # forget_gate: ft = σ (Wf ·[ht−1,xt] + bf)
            # forget_tensor = self.forget_gate.forward(concatenated_tensor)
            forget_tensor = self.sigmoid_forget[t].forward(forget_tensor)
            self.forget_tensor[t] = forget_tensor

            # output_gate: ot = σ (Wo ·[ht−1,xt] + bo)
            output_gate_tensor = self.sigmoid_output[t].forward(output_gate_tensor)
            self.output_gate_tensor[t] = output_gate_tensor

            # Ct = ft · Ct−1 + it · ˜Ct
            self.preCell_state_forward = update_tensor * cell_tensor + forget_tensor * self.cell_state[t]
            self.cell_state[t + 1] = self.preCell_state_forward

            # cell state to hidden state : ht = ot ·tanh(Ct)
            cell_to_hidden_tensor = self.tanh_hidden[t].forward(self.cell_state[t + 1])
            self.cell_to_hidden_tensor[t] = cell_to_hidden_tensor
            self.preHidden_state_forward = output_gate_tensor * cell_to_hidden_tensor
            self.hidden_state[t + 1] = self.preHidden_state_forward

            # output :yt = σ (Wy ·ht + by)
            output_tensor[t] = self.final_layer.forward(self.hidden_state[t + 1])
            self.final_X[t] = self.final_layer.X    # store the gradient w.r.t. weights
            output_tensor[t] = self.sigmoid_final[t].forward(output_tensor[t])

        return output_tensor

    def backward(self, error_tensor):
        """
        Resource which explained the math behind an LSTM,
        especially the back propagation!!!
        http://arunmallya.github.io/writeups/nn/lstm/index.html#/8
        """
        output = np.zeros((error_tensor.shape[0], self.input_size))
        gradient_first_layer = []
        gradient_final_layer = []

        self.preHidden_state_backward = np.zeros(self.hidden_size)
        self.preCell_state_backward = np.zeros(self.hidden_size)

        for t in range(error_tensor.shape[0]-1, -1, -1):
            # == compute gradient of output_gate and cell state ==
            # gradient w.r.t to output： δht = δyt * δσ + ht
            gradient_final = self.sigmoid_final[t].backward(error_tensor[t])
            self.final_layer.X = self.final_X[t]                                    # load the gradient w.r.t. weights

            gradient_hidden = self.final_layer.backward(gradient_final)
            gradient_final_layer.append(self.final_layer.get_gradient_weights())  # read gradient weights
            gradient_hidden += self.preHidden_state_backward

            # Forward pass: ht = ot ·tanh(Ct)
            # gradient w.r.t. output gate:  δot = δht * tanh(Ct)
            gradient_output_gate = gradient_hidden * self.cell_to_hidden_tensor[t]
            gradient_output_gate = self.sigmoid_output[t].backward(gradient_output_gate)

            # gradient w.r.t. next cell state:    δCt += δht * ot * (1 - tanh^2(Ct)
            gradient_cell_error = gradient_hidden * self.output_gate_tensor[t]
            gradient_cell_error = self.tanh_hidden[t].backward(gradient_cell_error)
            gradient_cell = gradient_cell_error + self.preCell_state_backward

            # == LSTM memory cell update ==
            # Forward pass: Ct = ft · Ct−1 + it · ˜Ct
            # gradient w.r.t. update gate:  δit = δCt * ˜Ct
            gradient_update = gradient_cell * self.cell_tensor[t]
            gradient_update = self.sigmoid_update[t].backward(gradient_update)

            # gradient w.r.t. temp cell state: δ˜Ct = δct * it
            gradient_cell_tensor = gradient_cell * self.update_tensor[t]
            gradient_cell_tensor = self.tanh_hidden[t].backward(gradient_cell_tensor)

            # gradient w.r.t. forget gate: δft = δCt * Ct-1
            gradient_forget = gradient_cell * self.cell_state[t]
            gradient_forget = self.sigmoid_forget[t].backward(gradient_forget)

            # gradient w.r.t. previous cell state: δCt-1 = δCt * ft
            self.preCell_state_backward = gradient_cell * self.forget_tensor[t]

            # == input and previous hidden state
            four_gradient_tensors = np.hstack((gradient_forget,
                                              gradient_update,
                                              gradient_cell_tensor,
                                              gradient_output_gate))
            self.first_layer.X = self.first_X[t]                                    # load the gradient w.r.t. weights
            gradient_input = self.first_layer.backward(four_gradient_tensors)
            gradient_first_layer.append(self.first_layer.get_gradient_weights())    # read gradient weights

            output[t] = gradient_input[self.hidden_size:]
            self.preHidden_state_backward = gradient_input[:self.hidden_size]

        # update
        gradient_first_layer = np.array(gradient_first_layer)
        gradient_final_layer = np.array(gradient_final_layer)

        gradient_first_layer = np.sum(gradient_first_layer, axis=0)
        gradient_final_layer = np.sum(gradient_final_layer, axis=0)
        self.first_layer.gradient_weights = gradient_first_layer
        self.final_layer.gradient_weights = gradient_final_layer

        if self._optimizer is not None:

            self.first_layer.weights = self._optimizer.calculate_update(self.first_layer.weights,
                                                                        gradient_first_layer)
            self.final_layer.weights = self._optimizer.calculate_update(self.final_layer.weights,
                                                                        gradient_final_layer)

        return output









