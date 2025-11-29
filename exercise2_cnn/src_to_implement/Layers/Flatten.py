import numpy as np
from Layers.Base import BaseLayer

class Flatten(BaseLayer):
    def __init__(self):
        """
        flatten layer
        """
        super().__init__()
        self.input_shape = None

    def forward(self, input_tensor):
        """
        reshape input tensor into 2D tensor
        """
        self.input_shape = input_tensor.shape
        batch_size = self.input_shape[0]
        return input_tensor.reshape(batch_size, -1)

    def backward(self, error_tensor):
        """
        reshape error tensor back to original input shape
        """
        return error_tensor.reshape(self.input_shape) 