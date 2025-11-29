import numpy as np
from Layers.Base import BaseLayer

class Dropout(BaseLayer):
    def __init__(self, probability):
        """
        inverted dropout
        
        args:
            probability: fraction of units to keep
        """
        super().__init__()
        self.probability = probability
        self.mask = None

    def forward(self, input_tensor):
        """
        forward pass
        
        args:
            input_tensor: input data
            
        returns:
            output tensor
        """
        if self.testing_phase:
            # no dropout
            return input_tensor
        else:
            self.mask = np.random.rand(*input_tensor.shape) < self.probability
            # inverted dropout
            output = input_tensor * self.mask / self.probability
            
            return output

    def backward(self, error_tensor):
        """
        backward pass
        
        args:
            error_tensor: gradient from next layer
            
        returns:
            gradient for previous layer
        """
        if self.testing_phase:
            # no dropout
            return error_tensor
        else:
            # inverted dropout
            return error_tensor * self.mask / self.probability 