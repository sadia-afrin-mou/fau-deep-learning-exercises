import numpy as np
from Layers.Base import BaseLayer

class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()
        self._input_tensor = None

    def forward(self, input_tensor):
        """
        ReLU activation
        args:
            input_tensor: input data
        returns:
            ReLU activated input
        """
        self._input_tensor = input_tensor
        return np.maximum(0, input_tensor)

    def backward(self, error_tensor):
        """
        compute gradients for ReLU
        args:
            error_tensor: gradient from next layer
        returns:
            gradient for last layer
        """
        return error_tensor * (self._input_tensor > 0) 