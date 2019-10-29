import numpy as np


class NeuralNetwork:
    def __init__(self, optimizer):
        self.optimizer = optimizer   # An optimizer object
        self.loss = []          # A list loss which will contain the loss value for each iteration after calling train.
        self.layers = []        # A list layers which will hold the architecture
        self.data_layer = None  # which will provide input data and labels
        self.loss_layer = None  # referring to the special layer providing loss and prediction.
        self.label_tensor = None

    """
    using input from the data_layer and passing it through all layers of the network.
    Note that the data layer provides an input tensor and a label tensor upon calling forward() on it.
    !! recursively calls forward on its layers passing the input-data !!
    
    @return  
    """
    def forward(self):
        input_tensor, self.label_tensor = self.data_layer.forward()
        tensor = np.copy(input_tensor)
        for it in self.layers:
            tensor = it.forward(tensor)
        loss = self.loss_layer.forward(tensor, self.label_tensor)
        return loss

    """
    starting from the loss layer passing it the label tensor for the current input 
    and propagating it back through the network. 
    """
    def backward(self):
        tensor = self.loss_layer.backward(self.label_tensor)
        for it in reversed(self.layers):
            tensor = it.backward(tensor)
        return tensor

    """
    which makes a deep copy of the neural networks optimizer and sets it for the layer by using its optimizer property. 
    Make sure you append this layer to the list layers afterwards.
    """
    def append_trainable_layer(self, layer):
        layer.set_optimizer(1e-03)
        self.layers.append(layer)

    """
    trains the network for iterations and stores the loss for each iteration.
    """
    def train(self, iterations):
        for i in range(iterations):
            loss = self.forward()
            self.loss.append(loss)
            self.backward()

    """
    propagates the input tensor through the network and returns the prediction of the last layer. 
    For classification tasks we typically query the probabilistic output of the SoftMax layer.
    """
    def test(self, input_tensor):
        tensor = np.copy(input_tensor)
        for it in self.layers:
            tensor = it.forward(tensor)
        return tensor

