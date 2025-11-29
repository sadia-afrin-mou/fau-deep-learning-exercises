import numpy as np
from Layers.Base import BaseLayer
from Layers.Helpers import compute_bn_gradients
import copy

class BatchNormalization(BaseLayer):
    def __init__(self, channels):
        """
        batch normalization layer
        
        args:
            channels: no of channels
        """
        super().__init__()
        self.trainable = True
        self.channels = channels
        
        # initialize weights (gamma) and bias (beta)
        self.weights = np.ones(channels)
        self.bias = np.zeros(channels)

        self._gradient_weights = None
        self._gradient_bias = None
        
        self._optimizer = None
        
        # moving averages for test phase
        self.moving_mean = None
        self.moving_var = None
        self.alpha = 0.8  # momentum
        
        # to be used in backward method
        self.input_tensor = None
        self.mean = None
        self.var = None
        self.normalized_input = None
        
        self.eps = np.finfo(float).eps
        
    def initialize(self, weights_initializer=None, bias_initializer=None):
        """
        ignores other initializers and always sets weights to 1 and bias to 0
        """
        self.weights = np.ones(self.channels)
        self.bias = np.zeros(self.channels)
        
    @property
    def optimizer(self):
        return self._optimizer
        
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        
    @property
    def gradient_weights(self):
        return self._gradient_weights
        
    @property
    def gradient_bias(self):
        return self._gradient_bias
        
    def reformat(self, tensor):
        """
        reformat tensor between 2D and 4D
        
        args:
            tensor: input tensor to reformat
            
        returns:
            reformatted tensor
        """
        if tensor.ndim == 4:
            # 4D to 2D: (batch, channels, height, width) to (batch*height*width, channels)
            batch_size, channels, height, width = tensor.shape
            return tensor.transpose(0, 2, 3, 1).reshape(batch_size * height * width, channels)
        elif tensor.ndim == 2:
            # 2D to 4D: (batch*height*width, channels) to (batch, channels, height, width)
            batch_size, channels, height, width = self.input_shape
            return tensor.reshape(batch_size, height, width, channels).transpose(0, 3, 1, 2)
        else:
            raise ValueError("tensor must be 2D or 4D") 
    
    def forward(self, input_tensor):
        """
        forward pass
        
        args:
            input_tensor: input data
            
        returns:
            normalized output
        """
        self.input_tensor = input_tensor
        
        if input_tensor.ndim == 4:
            # conv case
            self.input_shape = input_tensor.shape
            # reformatted to 2D
            input_2d = self.reformat(input_tensor)
        else:
            # fc case
            input_2d = input_tensor
            
        if self.testing_phase:
            # moving averages
            if self.moving_mean is None or self.moving_var is None:
                self.mean = np.mean(input_2d, axis=0)
                self.var = np.var(input_2d, axis=0)
            else:
                self.mean = self.moving_mean
                self.var = self.moving_var
        else:
            # batch statistics
            self.mean = np.mean(input_2d, axis=0)
            self.var = np.var(input_2d, axis=0)
            
            if self.moving_mean is None:
                self.moving_mean = self.mean.copy()
                self.moving_var = self.var.copy()
            else:
                self.moving_mean = self.alpha * self.moving_mean + (1 - self.alpha) * self.mean
                self.moving_var = self.alpha * self.moving_var + (1 - self.alpha) * self.var
        
        # normalize
        self.normalized_input = (input_2d - self.mean) / np.sqrt(self.var + self.eps)
        
        output_2d = self.weights * self.normalized_input + self.bias
        
        # back to original shape
        if input_tensor.ndim == 4:
            output = self.reformat(output_2d)
        else:
            output = output_2d
            
        return output
    
    def backward(self, error_tensor):
        """
        backward pass
        
        args:
            error_tensor: gradient from next layer
            
        returns:
            gradient for previous layer
        """
        if error_tensor.ndim == 4:
            # conv case
            error_2d = self.reformat(error_tensor)
            input_2d = self.reformat(self.input_tensor)
        else:
            # fc case
            error_2d = error_tensor
            input_2d = self.input_tensor
            
        # gradients w.r.t. weights and bias
        self._gradient_weights = np.sum(error_2d * self.normalized_input, axis=0)
        self._gradient_bias = np.sum(error_2d, axis=0)
        
        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
            self.bias = self._optimizer.calculate_update(self.bias, self._gradient_bias)
        
        # gradient w.r.t. input
        gradient_input_2d = compute_bn_gradients(error_2d, input_2d, self.weights, 
                                                self.mean, self.var, self.eps)
        
        # back to original
        if error_tensor.ndim == 4:
            gradient_input = self.reformat(gradient_input_2d)
        else:
            gradient_input = gradient_input_2d
            
        return gradient_input
