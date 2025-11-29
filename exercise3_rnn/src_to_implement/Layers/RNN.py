import numpy as np
from Layers.Base import BaseLayer
from Layers.FullyConnected import FullyConnected
from Layers.TanH import TanH
import copy

class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Elman RNN layer
        
        args:
            input_size: dimension of input tensor
            hidden_size: dimension of hidden state
            output_size: dimension of output tensor
        """
        self.trainable = True
        self.testing_phase = False
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # hidden state
        self.hidden_state = np.zeros((1, hidden_size))
        
        # memorize state
        self._memorize = False
        
        # sub layers
        self.fc_hidden = FullyConnected(input_size + hidden_size, hidden_size)
        self.tanh = TanH()
        self.fc_output = FullyConnected(hidden_size, output_size)
        
        # to be used in backward
        self.input_sequence = None
        self.hidden_states = None
        self.hidden_states_before_activation = None
        self.hidden_states_after_activation = None
        self.output_sequence = None
        
        # inputs to store for fc layer
        self.fc_hidden_inputs = []
        self.fc_output_inputs = []
        
        # gradients
        self.gradient_weights_fc_hidden = np.zeros_like(self.fc_hidden.weights)
        self.gradient_weights_fc_output = np.zeros_like(self.fc_output.weights)
        
        self._optimizer = None
        
    @property
    def memorize(self):
        """get memorize state"""
        return self._memorize
        
    @memorize.setter
    def memorize(self, value):
        """set memorize state"""
        self._memorize = value
        if not value:
            # reset hidden state
            self.hidden_state = np.zeros((1, self.hidden_size))
    
    @property
    def weights(self):
        """get weights"""
        return self.fc_hidden.weights
        
    @weights.setter
    def weights(self, weights):
        """set weights"""
        self.fc_hidden.weights = weights
        
    @property
    def gradient_weights(self):
        """get gradient weights"""
        return self.gradient_weights_fc_hidden
        
    @property
    def optimizer(self):
        """get optimizer"""
        return self._optimizer
        
    @optimizer.setter
    def optimizer(self, optimizer):
        """set optimizer"""
        self._optimizer = optimizer
        self.fc_hidden.optimizer = copy.deepcopy(optimizer)
        self.fc_output.optimizer = copy.deepcopy(optimizer)
    
    def initialize(self, weights_initializer, bias_initializer):
        self.fc_hidden.initialize(weights_initializer, bias_initializer)
        self.fc_output.initialize(weights_initializer, bias_initializer)
        
    def calculate_regularization_loss(self):
        """
        regularization loss - all sub layers
        
        returns:
            total regularization loss
        """
        total_loss = 0
        
        # fc_hidden
        if (hasattr(self.fc_hidden, 'optimizer') and 
            self.fc_hidden.optimizer is not None and
            hasattr(self.fc_hidden.optimizer, 'regularizer') and
            self.fc_hidden.optimizer.regularizer is not None):
            
            regularizer = self.fc_hidden.optimizer.regularizer
            total_loss += regularizer.norm(self.fc_hidden.weights)
            
        # fc_output
        if (hasattr(self.fc_output, 'optimizer') and 
            self.fc_output.optimizer is not None and
            hasattr(self.fc_output.optimizer, 'regularizer') and
            self.fc_output.optimizer.regularizer is not None):
            
            regularizer = self.fc_output.optimizer.regularizer
            total_loss += regularizer.norm(self.fc_output.weights)
            
        return total_loss
    
    def forward(self, input_tensor):
        """
        forward pass
        
        args:
            input_tensor: input sequence (sequence_length, input_size)
                         batch dimension is time steps
            
        returns:
            output sequence (sequence_length, output_size)
        """
        sequence_length = input_tensor.shape[0]
        self.input_sequence = input_tensor.copy()
        
        # storage for states
        self.hidden_states = np.zeros((sequence_length + 1, self.hidden_size))
        self.hidden_states_before_activation = np.zeros((sequence_length, self.hidden_size))
        self.hidden_states_after_activation = np.zeros((sequence_length, self.hidden_size))
        
        # reset input lists
        self.fc_hidden_inputs = []
        self.fc_output_inputs = []
        
        # initial hidden state
        if not self.memorize:
            self.hidden_state = np.zeros((1, self.hidden_size))
        self.hidden_states[0] = self.hidden_state.copy()
        
        # output sequence
        self.output_sequence = np.zeros((sequence_length, self.output_size))
        
        # forward pass
        for t in range(sequence_length):
            input_t = input_tensor[t:t+1]
            
            # concatenate hidden state and input
            combined_input = np.hstack([self.hidden_states[t:t+1], input_t])
            self.fc_hidden_inputs.append(combined_input.copy())
            
            # hidden state
            hidden_linear = self.fc_hidden.forward(combined_input)
            self.hidden_states_before_activation[t] = hidden_linear.copy()
            
            hidden_activated = self.tanh.forward(hidden_linear)
            self.hidden_states_after_activation[t] = hidden_activated.copy()
            
            # store hidden state
            self.hidden_states[t+1] = hidden_activated.copy()
            
            # store input for fc_output
            self.fc_output_inputs.append(hidden_activated.copy())
            
            # output
            output_t = self.fc_output.forward(hidden_activated)
            self.output_sequence[t] = output_t.copy()
        
        # memorize state
        if self.memorize:
            self.hidden_state = self.hidden_states[-1:].copy()
            
        return self.output_sequence
    
    def backward(self, error_tensor):
        """
        backward pass
        
        args:
            error_tensor: gradient from next layer (sequence_length, output_size)
            
        returns:
            gradient for previous layer (sequence_length, input_size)
        """
        sequence_length = error_tensor.shape[0]
        
        # gradients
        input_gradient = np.zeros((sequence_length, self.input_size))
        hidden_gradient = np.zeros((1, self.hidden_size))
        
        # reset gradients
        self.gradient_weights_fc_hidden = np.zeros_like(self.fc_hidden.weights)
        self.gradient_weights_fc_output = np.zeros_like(self.fc_output.weights)
        
        # tanh gradients
        tanh_gradients = 1 - self.hidden_states_after_activation**2
        
        # backward loop
        for t in reversed(range(sequence_length)):
            output_error = error_tensor[t:t+1]

            # backward for fc_output
            self.fc_output._input_tensor = self.fc_output_inputs[t]
            hidden_error_from_fc_output = self.fc_output.backward(output_error)
            
            # gradients from fc_output
            self.gradient_weights_fc_output += self.fc_output._gradient_weights
            
            # future hidden error
            total_hidden_error = hidden_error_from_fc_output + hidden_gradient
            
            # tanh gradient
            hidden_error_before_activation = total_hidden_error * tanh_gradients[t:t+1]
            
            # backward for fc_hidden
            self.fc_hidden._input_tensor = self.fc_hidden_inputs[t]
            combined_gradient = self.fc_hidden.backward(hidden_error_before_activation)
            
            # gradients from fc_hidden
            self.gradient_weights_fc_hidden += self.fc_hidden._gradient_weights
            
            # split gradient into hidden and input
            hidden_gradient = combined_gradient[:, :self.hidden_size]
            input_gradient[t] = combined_gradient[0, self.hidden_size:]
        
        # optimizer
        if self.optimizer is not None:
            self.fc_output.weights = self.optimizer.calculate_update(
                self.fc_output.weights, 
                self.gradient_weights_fc_output
            )
            
            self.fc_hidden.weights = self.optimizer.calculate_update(
                self.fc_hidden.weights, 
                self.gradient_weights_fc_hidden
            )
        
        return input_gradient
