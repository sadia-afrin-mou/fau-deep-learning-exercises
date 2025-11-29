import numpy as np
from Layers.Base import BaseLayer

class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        self._prediction_tensor = None

    def forward(self, input_tensor):
        """
        softmax activation
        
        args:
            input_tensor: input data
            
        returns:
            probability distribution
        """
        input_shifted = input_tensor - np.max(input_tensor, axis=1, keepdims=True)
        
        input_exp = np.exp(input_shifted)
        
        self._prediction_tensor = input_exp / np.sum(input_exp, axis=1, keepdims=True)
        
        return self._prediction_tensor

    def backward(self, error_tensor):
        """
        gradients for softmax
        
        args:
            error_tensor: gradient from next layer
            
        returns:
            gradient for last layer
        """
        return self._prediction_tensor * (error_tensor - np.sum(error_tensor * self._prediction_tensor, axis=1, keepdims=True)) 