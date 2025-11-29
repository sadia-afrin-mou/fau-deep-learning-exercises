import numpy as np
from Layers.Base import BaseLayer

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        """
        fully connected layer
        
        args:
            input_size: number of input features
            output_size: number of neurons in this layer
        """
        super().__init__()
        self.trainable = True
        
        self.weights = np.random.uniform(0, 1, (input_size + 1, output_size))
        
        self._optimizer = None
        self._input_tensor = None
        self._gradient_weights = None

    @property
    def optimizer(self):
        """getter for the optimizer"""
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        """setter for the optimizer"""
        self._optimizer = optimizer

    @property
    def gradient_weights(self):
        """getter for the _gradient_weights"""
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        """setter for the _gradient_weights"""
        self._gradient_weights = gradient_weights

    def forward(self, input_tensor):
        """
        forward pass
        
        args:
            input_tensor: input data (batch_size, input_size)
            
        returns:
            output tensor (batch_size, output_size)
        """
        self._input_tensor = input_tensor
        
        batch_size = input_tensor.shape[0]
        bias_tensor = np.ones((batch_size, 1))
        input_with_bias = np.hstack((input_tensor, bias_tensor))
        
        return input_with_bias @ self.weights

    def backward(self, error_tensor):
        """
        backward pass
        
        args:
            error_tensor: gradient from next layer
            
        returns:
            gradient for last layer
        """
        batch_size = self._input_tensor.shape[0]
        bias_tensor = np.ones((batch_size, 1))
        input_with_bias = np.hstack((self._input_tensor, bias_tensor))
        
        # gradient w.r.t weights
        self._gradient_weights = input_with_bias.T @ error_tensor
        
        # weights update
        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
        
        # gradient for last layer, bias term is excluded
        return error_tensor @ self.weights.T[:, :-1] 