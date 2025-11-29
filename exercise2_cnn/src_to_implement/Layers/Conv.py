import numpy as np
from scipy import signal
from scipy.signal import correlate2d, convolve2d
from Layers.Base import BaseLayer
import copy

class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        """
        convolutional layer
        
        args:
            stride_shape: integer or tuple of integers for stride
            convolution_shape: shape of kernel (channels, *dims)
            num_kernels: number of kernels
        """
        super().__init__()
        self.trainable = True
        
        if isinstance(stride_shape, int):
            stride_shape = (stride_shape, stride_shape)
        elif len(stride_shape) == 1:
            stride_shape = (stride_shape[0], stride_shape[0])
        self.stride_shape = stride_shape
        
        self.weights = np.random.uniform(size=(num_kernels, *convolution_shape))
        self.is_conv2d = (len(convolution_shape) == 3)
        if self.is_conv2d:
            self.convolution_shape = convolution_shape
        else:
            self.convolution_shape = (*convolution_shape, 1)
            self.weights = self.weights[:, :, :, np.newaxis]
            
        self.num_kernels = num_kernels
        self.bias = np.random.uniform(size=(num_kernels,))
        
        self._gradient_weights = None
        self._gradient_bias = None
        self._optimizer = None
        
        self.input_tensor = None
        self.output_shape = None

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self._optimizer.weights = copy.deepcopy(optimizer)
        self._optimizer.bias = copy.deepcopy(optimizer)

    def initialize(self, weights_initializer, bias_initializer):
        """
        initialize weights and bias
        
        args:
            weights_initializer: initializer for weights
            bias_initializer: initializer for bias
        """
        fan_in = np.prod(self.convolution_shape)
        fan_out = np.prod(self.convolution_shape[1:]) * self.num_kernels
        self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, 1, self.num_kernels)

    def forward(self, input_tensor):
        """
        forward pass
        
        args:
            input_tensor: input data of shape (batch, channels, *dims)
            
        returns:
            output tensor after convolution
        """
        self.input_tensor = input_tensor
        
        # 1D convolution case
        if input_tensor.ndim == 3:
            input_tensor = input_tensor[:, :, :, np.newaxis]
        
        batch_size, channels, height, width = input_tensor.shape
        
        # padded input
        padded_input = np.zeros((
            batch_size, 
            channels, 
            height + self.convolution_shape[1] - 1, 
            width + self.convolution_shape[2] - 1
        ))
        
        # padding based on kernel size
        pad_y_adjust = int(self.convolution_shape[1]//2 == self.convolution_shape[1]/2)
        pad_x_adjust = int(self.convolution_shape[2]//2 == self.convolution_shape[2]/2)
        
        if self.convolution_shape[1]//2 == 0 and self.convolution_shape[2]//2 == 0:
            padded_input = input_tensor
        else:
            padded_input[:, :, 
                        (self.convolution_shape[1]//2):-(self.convolution_shape[1]//2)+pad_y_adjust, 
                        (self.convolution_shape[2]//2):-(self.convolution_shape[2]//2)+pad_x_adjust] = input_tensor
            
        # output dimensions
        out_height = int(np.ceil((padded_input.shape[2] - self.convolution_shape[1] + 1) / self.stride_shape[0]))
        out_width = int(np.ceil((padded_input.shape[3] - self.convolution_shape[2] + 1) / self.stride_shape[1]))
            
        # initialize output tensor
        output_tensor = np.zeros((batch_size, self.num_kernels, out_height, out_width))
        self.output_shape = output_tensor.shape
        
        # convolution operation
        for b in range(batch_size):
            for k in range(self.num_kernels):
                for h in range(out_height):
                    for w in range(out_width):
                        # window boundaries
                        h_start = h * self.stride_shape[0]
                        h_end = h_start + self.convolution_shape[1]
                        w_start = w * self.stride_shape[1]
                        w_end = w_start + self.convolution_shape[2]
                        
                        # check bounds
                        if (h_end <= padded_input.shape[2]) and (w_end <= padded_input.shape[3]):
                            # current window
                            window = padded_input[b, :, h_start:h_end, w_start:w_end]
                            
                            # convolution operation
                            output_tensor[b, k, h, w] = np.sum(window * self.weights[k]) + self.bias[k]
                        else:
                            output_tensor[b, k, h, w] = 0

        # 1D convolution output
        if not self.is_conv2d:
            output_tensor = output_tensor.squeeze(axis=3)
            
        return output_tensor

    def backward(self, error_tensor):
        """
        backward pass
        
        args:
            error_tensor: gradient from next layer
            
        returns:
            gradient for previous layer
        """
        reshaped_error_tensor = error_tensor.reshape(self.output_shape)
        
        # 1D convolution
        if not self.is_conv2d:
            self.input_tensor = self.input_tensor[:, :, :, np.newaxis]
            
        batch_size, channels, height, width = self.input_tensor.shape
        
        # initialize gradient tensors
        upsampled_error_tensor = np.zeros((batch_size, self.num_kernels, height, width))
        input_gradient = np.zeros(self.input_tensor.shape)
        self._gradient_bias = np.zeros(self.num_kernels)
        self._gradient_weights = np.zeros(self.weights.shape)
        
        # padding dimensions
        pad_height = int(np.floor(self.convolution_shape[1] / 2))
        pad_width = int(np.floor(self.convolution_shape[2] / 2))
        
        # padded input for gradient calculation
        padded_input = np.zeros((
            batch_size, 
            channels, 
            height + self.convolution_shape[1] - 1,
            width + self.convolution_shape[2] - 1
        ))

        # gradient calculation
        for b in range(batch_size):
            for k in range(self.num_kernels):
                # gradient w.r.t bias
                self._gradient_bias[k] += np.sum(reshaped_error_tensor[b, k])

                # upsample error tensor
                for h in range(reshaped_error_tensor.shape[2]):
                    for w in range(reshaped_error_tensor.shape[3]):
                        # error boundaries
                        h_pos = h * self.stride_shape[0]
                        w_pos = w * self.stride_shape[1]
                        
                        # upsample operation
                        upsampled_error_tensor[b, k, h_pos, w_pos] = reshaped_error_tensor[b, k, h, w]

                # gradient w.r.t input
                for ch in range(channels):
                    input_gradient[b, ch] += convolve2d(
                        upsampled_error_tensor[b, k], 
                        self.weights[k, ch], 
                        'same'
                    )

            # padded input for weight gradients
            for ch in range(channels):
                for h in range(padded_input.shape[2]):
                    for w in range(padded_input.shape[3]):
                        # padding boundaries
                        if (h > pad_height - 1) and (h < height + pad_height):
                            if (w > pad_width - 1) and (w < width + pad_width):
                                # padding operation
                                padded_input[b, ch, h, w] = self.input_tensor[b, ch, h - pad_height, w - pad_width]

            # gradient w.r.t weights
            for k in range(self.num_kernels):
                for ch in range(channels):
                    self._gradient_weights[k, ch] += correlate2d(
                        padded_input[b, ch], 
                        upsampled_error_tensor[b, k], 
                        'valid'
                    )

        # update weights using optimizer
        if self._optimizer is not None:
            self.weights = self._optimizer.weights.calculate_update(self.weights, self._gradient_weights)
            self.bias = self._optimizer.bias.calculate_update(self.bias, self._gradient_bias)

        # 1D convolution output
        if not self.is_conv2d:
            input_gradient = input_gradient.squeeze(axis=3)
            
        return input_gradient