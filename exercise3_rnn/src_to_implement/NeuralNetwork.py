import copy
import numpy as np
import pickle
import os

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

    def __getstate__(self):
        """
        Custom method for pickling - excludes data_layer
        """
        state = self.__dict__.copy()
        # Remove the data_layer as it's a generator and can't be pickled
        state['data_layer'] = None
        return state

    def __setstate__(self, state):
        """
        Custom method for unpickling - initializes data_layer to None
        """
        self.__dict__.update(state)
        self.data_layer = None

    @property
    def phase(self):
        """
        Get the current phase of the network
        """
        return not self.layers[0].testing_phase if self.layers else True

    @phase.setter
    def phase(self, phase):
        """
        Set the phase of all layers in the network
        
        args:
            phase: True for training, False for testing
        """
        for layer in self.layers:
            layer.testing_phase = not phase

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
            
        data_loss = self.loss_layer.forward(output, self.label_tensor)
        
        # regularization loss
        regularization_loss = 0
        for layer in self.layers:
            if layer.trainable and hasattr(layer, 'optimizer'):
                optimizer = layer.optimizer
                if optimizer is not None and hasattr(optimizer, 'regularizer') and optimizer.regularizer is not None:
                    # regularization for weights
                    if hasattr(layer, 'weights') and layer.weights is not None:
                        regularization_loss += optimizer.regularizer.norm(layer.weights)
                    
                    # regularization for bias
                    if hasattr(layer, 'bias') and layer.bias is not None:
                        if hasattr(layer, '_gradient_bias'):
                            regularization_loss += optimizer.regularizer.norm(layer.bias)
        
        return data_loss + regularization_loss

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
        self.phase = True  # Set training phase
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
        self.phase = False  # Set testing phase
        output = input_tensor
        for layer in self.layers:
            output = layer.forward(output)
        return output


# Standalone functions for saving and loading networks
def save(filename, net):
    """
    Save a neural network to a file using pickle
    
    args:
        filename: path to save the network
        net: NeuralNetwork instance to save
    """
    # Create directory if it doesn't exist
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    with open(filename, 'wb') as f:
        pickle.dump(net, f)

def load(filename, data_layer):
    """
    Load a neural network from a file and set its data layer
    
    args:
        filename: path to load the network from
        data_layer: data layer to set after loading
        
    returns:
        loaded NeuralNetwork instance
    """
    with open(filename, 'rb') as f:
        net = pickle.load(f)
    net.data_layer = data_layer
    return net