import copy
import numpy as np

class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer=None, bias_initializer=None):
        """
        neural network
        
        args:
            optimizer: optimizer instance
            weights_initializer: weights initializer
            bias_initializer: bias initializer
        """
        self.optimizer = optimizer
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.label_tensor = None

    def forward(self):
        """
        forward pass
        
        returns:
            network predictions
        """
        input_tensor, self.label_tensor = self.data_layer.next()
        
        output = input_tensor
        for layer in self.layers:
            output = layer.forward(output)
            
        return self.loss_layer.forward(output, self.label_tensor)

    def backward(self):
        """
        backward pass
        """
        error_tensor = self.loss_layer.backward(self.label_tensor)
        
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def append_layer(self, layer):
        """
        append layer to layers list
        
        args:
            layer: layer to add to network
        """
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
            if hasattr(layer, 'initialize') and self.weights_initializer and self.bias_initializer:
                layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)

    def train(self, iterations):
        """
        train network as per given iterations
        
        args:
            iterations: training iterations
        """
        for _ in range(iterations):
            loss_value = self.forward()
            self.backward()
            self.loss.append(loss_value)

    def test(self, input_tensor):
        """
        test network on test data
        
        args:
            input_tensor: input data
            
        returns:
            predictions
        """
        output = input_tensor
        for layer in self.layers:
            output = layer.forward(output)
        return output