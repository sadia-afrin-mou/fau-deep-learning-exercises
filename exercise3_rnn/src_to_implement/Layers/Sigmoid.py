import numpy as np
from Layers.Base import BaseLayer

class Sigmoid(BaseLayer):
    def __init__(self):
        super().__init__()
        self.activations = None
    
    def forward(self, input_tensor):
        # 1 / (1 + exp(-x))
        self.activations = 1 / (1 + np.exp(-input_tensor))
        return self.activations

    def backward(self, error_tensor):
        # sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        return error_tensor * self.activations * (1 - self.activations)
 