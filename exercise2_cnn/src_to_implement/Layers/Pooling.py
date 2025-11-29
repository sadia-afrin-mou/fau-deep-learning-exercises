import numpy as np
from Layers.Base import BaseLayer

class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        """
        pooling layer
        
        args:
            stride_shape: tuple of integers for stride
            pooling_shape: tuple of integers for pooling window size
        """
        super().__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        
        self.input_tensor = None
        self.max_indices = None

    def forward(self, input_tensor):
        """
        forward pass
        
        args:
            input_tensor: input data (batch, channels, *dims)
            
        returns:
            output tensor after max pooling
        """
        self.input_tensor = input_tensor
        
        batch_size, channels, height, width = input_tensor.shape
        
        # output dimensions
        out_height = (height - self.pooling_shape[0]) // self.stride_shape[0] + 1
        out_width = (width - self.pooling_shape[1]) // self.stride_shape[1] + 1
        
        # output tensor and indices for backward pass    
        output = np.zeros((batch_size, channels, out_height, out_width))
        self.max_indices = np.zeros((batch_size, channels, out_height, out_width, 2), dtype=int)
        
        # max pooling
        for b in range(batch_size):
            for c in range(channels):
                for h in range(out_height):
                    for w in range(out_width):
                        # window boundaries
                        h_start = h * self.stride_shape[0]
                        h_end = h_start + self.pooling_shape[0]
                        w_start = w * self.stride_shape[1]
                        w_end = w_start + self.pooling_shape[1]
                        
                        # current window
                        window = input_tensor[b, c, h_start:h_end, w_start:w_end]
                        
                        # max value and its position
                        max_val = np.max(window)
                        max_pos = np.unravel_index(np.argmax(window), window.shape)
                        
                        # storing max value
                        output[b, c, h, w] = max_val
                        
                        # storing position of max
                        self.max_indices[b, c, h, w] = [h_start + max_pos[0], 
                                                       w_start + max_pos[1]]
        
        return output

    def backward(self, error_tensor):
        """
        backward pass
        
        args:
            error_tensor: gradient from next layer (batch, channels, *dims)
            
        returns:
            gradient for previous layer
        """
        batch_size, channels, out_height, out_width = error_tensor.shape
        
        # gradient tensor
        input_gradient = np.zeros_like(self.input_tensor)
        
        # distribute gradients
        for b in range(batch_size):
            for c in range(channels):
                for h in range(out_height):
                    for w in range(out_width):
                        # position of max
                        h_max, w_max = self.max_indices[b, c, h, w]
                        
                        # put gradient to position
                        input_gradient[b, c, h_max, w_max] += error_tensor[b, c, h, w]
        
        return input_gradient