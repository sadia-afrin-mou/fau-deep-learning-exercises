import numpy as np
from Layers.Base import BaseLayer
from Layers.FullyConnected import FullyConnected
from Layers.TanH import TanH
from Layers.Sigmoid import Sigmoid
import copy

class LSTM(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        """
        LSTM layer
        
        args:
            input_size: dimension of input tensor
            hidden_size: dimension of hidden state and cell state
            output_size: dimension of output tensor
        """
        self.trainable = True
        self.testing_phase = False
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # hidden state and cell state
        self.hidden_state = np.zeros((1, hidden_size))
        self.cell_state = np.zeros((1, hidden_size))
        
        # memorize state
        self._memorize = False
        
        # fc layer for all 4 gates (forget, input, candidate, output)
        self.fc_gates = FullyConnected(input_size + hidden_size, 4 * hidden_size)
        
        # fc layer for hidden to output
        self.fc_output = FullyConnected(hidden_size, output_size)
        
        # activation functions
        self.sigmoid = Sigmoid()
        self.tanh = TanH()
        
        # to be used in backward
        self.input_sequence = None
        self.hidden_states = None
        self.cell_states = None
        self.gate_outputs = None
        self.forget_gates = None
        self.input_gates = None
        self.candidate_values = None
        self.output_gates = None
        self.output_sequence = None
        
        # inputs to store for fc layer
        self.fc_gates_inputs = []
        self.fc_output_inputs = []
        
        # gradients
        self.gradient_weights_fc_gates = np.zeros_like(self.fc_gates.weights)
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
            # reset hidden and cell states
            self.hidden_state = np.zeros((1, self.hidden_size))
            self.cell_state = np.zeros((1, self.hidden_size))
    
    @property
    def weights(self):
        """get weights"""
        return self.fc_gates.weights
        
    @weights.setter
    def weights(self, weights):
        """set weights"""
        self.fc_gates.weights = weights
        
    @property
    def gradient_weights(self):
        """get gradient weights"""
        return self.gradient_weights_fc_gates
        
    @property
    def optimizer(self):
        """get optimizer"""
        return self._optimizer
        
    @optimizer.setter
    def optimizer(self, optimizer):
        """set optimizer"""
        self._optimizer = optimizer
        self.fc_gates.optimizer = copy.deepcopy(optimizer)
        self.fc_output.optimizer = copy.deepcopy(optimizer)
    
    def initialize(self, weights_initializer, bias_initializer):
        """initialize weights and bias"""
        self.fc_gates.initialize(weights_initializer, bias_initializer)
        self.fc_output.initialize(weights_initializer, bias_initializer)
        
    def calculate_regularization_loss(self):
        """
        regularization loss - all sub layers
        
        returns:
            total regularization loss
        """
        total_loss = 0
        
        # fc_gates
        if (hasattr(self.fc_gates, 'optimizer') and 
            self.fc_gates.optimizer is not None and
            hasattr(self.fc_gates.optimizer, 'regularizer') and
            self.fc_gates.optimizer.regularizer is not None):
            
            regularizer = self.fc_gates.optimizer.regularizer
            total_loss += regularizer.norm(self.fc_gates.weights)
            
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
        
        # storage for states and gates
        self.hidden_states = np.zeros((sequence_length + 1, self.hidden_size))
        self.cell_states = np.zeros((sequence_length + 1, self.hidden_size))
        self.gate_outputs = np.zeros((sequence_length, 4 * self.hidden_size))
        self.forget_gates = np.zeros((sequence_length, self.hidden_size))
        self.input_gates = np.zeros((sequence_length, self.hidden_size))
        self.candidate_values = np.zeros((sequence_length, self.hidden_size))
        self.output_gates = np.zeros((sequence_length, self.hidden_size))
        
        # reset input lists
        self.fc_gates_inputs = []
        self.fc_output_inputs = []
        
        # initial states
        if not self._memorize:
            self.hidden_state = np.zeros((1, self.hidden_size))
            self.cell_state = np.zeros((1, self.hidden_size))
            
        self.hidden_states[0] = self.hidden_state.copy()
        self.cell_states[0] = self.cell_state.copy()
        
        # output sequence
        self.output_sequence = np.zeros((sequence_length, self.output_size))
        
        # forward pass
        for t in range(sequence_length):
            input_t = input_tensor[t:t+1]
            
            # concatenate input and previous hidden state
            combined_input = np.hstack([self.hidden_states[t:t+1], input_t])
            self.fc_gates_inputs.append(combined_input.copy())
            
            # all gates
            gates_output = self.fc_gates.forward(combined_input)
            self.gate_outputs[t] = gates_output.copy()
            
            # 4 gates
            forget_gate_input = gates_output[:, :self.hidden_size]
            input_gate_input = gates_output[:, self.hidden_size:2*self.hidden_size]
            candidate_input = gates_output[:, 2*self.hidden_size:3*self.hidden_size]
            output_gate_input = gates_output[:, 3*self.hidden_size:]
            
            # activations
            forget_gate = self.sigmoid.forward(forget_gate_input)
            input_gate = self.sigmoid.forward(input_gate_input)
            candidate_values = self.tanh.forward(candidate_input)
            output_gate = self.sigmoid.forward(output_gate_input)
            
            # gate values
            self.forget_gates[t] = forget_gate.copy()
            self.input_gates[t] = input_gate.copy()
            self.candidate_values[t] = candidate_values.copy()
            self.output_gates[t] = output_gate.copy()
            
            # cell state: C_t = f_t * C_{t-1} + i_t * C̃_t
            new_cell_state = (forget_gate * self.cell_states[t:t+1] + 
                             input_gate * candidate_values)
            self.cell_states[t+1:t+2] = new_cell_state
            
            # hidden state: h_t = o_t * tanh(C_t)
            cell_tanh = self.tanh.forward(new_cell_state)
            new_hidden_state = output_gate * cell_tanh
            self.hidden_states[t+1:t+2] = new_hidden_state
            
            # store input for fc_output
            self.fc_output_inputs.append(new_hidden_state.copy())
            
            # output
            output_t = self.fc_output.forward(new_hidden_state)
            self.output_sequence[t] = output_t.copy()
        
        # memorize state
        if self._memorize:
            self.hidden_state = self.hidden_states[-1:].copy()
            self.cell_state = self.cell_states[-1:].copy()
            
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
        cell_gradient = np.zeros((1, self.hidden_size))
        
        # reset gradients
        self.gradient_weights_fc_gates = np.zeros_like(self.fc_gates.weights)
        self.gradient_weights_fc_output = np.zeros_like(self.fc_output.weights)
        
        # backward loop
        for t in reversed(range(sequence_length)):
            output_error = error_tensor[t:t+1]
            
            # input tensor for fc_output
            self.fc_output._input_tensor = self.fc_output_inputs[t]
            
            # backward for fc_output
            hidden_error_from_fc_output = self.fc_output.backward(output_error)
            
            # gradients from fc_output
            self.gradient_weights_fc_output += self.fc_output._gradient_weights
            
            # future hidden error
            total_hidden_error = hidden_error_from_fc_output + hidden_gradient
            
            # current cell state
            current_cell = self.cell_states[t+1:t+2]
            
            # gradient w.r.t. output gate
            tanh_cell = np.tanh(current_cell)
            output_gate_error = total_hidden_error * tanh_cell
            
            # gradient w.r.t. cell state from hidden state
            tanh_derivative = 1 - tanh_cell**2
            cell_error_from_hidden = (total_hidden_error * self.output_gates[t:t+1] * 
                                    tanh_derivative)
            
            # future cell error
            total_cell_error = cell_error_from_hidden + cell_gradient
            
            # gradient w.r.t. forget gate
            prev_cell = self.cell_states[t:t+1]
            forget_gate_error = total_cell_error * prev_cell
            
            # gradient w.r.t. input gate
            input_gate_error = total_cell_error * self.candidate_values[t:t+1]
            
            # gradient w.r.t. candidate values
            candidate_error = total_cell_error * self.input_gates[t:t+1]
            
            # sigmoid derivative: σ(x) * (1 - σ(x))
            forget_sigmoid = self.forget_gates[t:t+1]
            forget_gate_linear_error = forget_gate_error * forget_sigmoid * (1 - forget_sigmoid)
            
            input_sigmoid = self.input_gates[t:t+1]
            input_gate_linear_error = input_gate_error * input_sigmoid * (1 - input_sigmoid)
            
            # tanh derivative: 1 - tanh²(x)
            candidate_tanh = self.candidate_values[t:t+1]
            candidate_linear_error = candidate_error * (1 - candidate_tanh**2)
            
            output_sigmoid = self.output_gates[t:t+1]
            output_gate_linear_error = output_gate_error * output_sigmoid * (1 - output_sigmoid)
            
            # combined gate errors
            combined_gate_error = np.hstack([
                forget_gate_linear_error,
                input_gate_linear_error,
                candidate_linear_error,
                output_gate_linear_error
            ])
            
            # input tensor for gates
            self.fc_gates._input_tensor = self.fc_gates_inputs[t]
            
            # backward for gates
            combined_input_gradient = self.fc_gates.backward(combined_gate_error)
            
            # gradients from gates
            self.gradient_weights_fc_gates += self.fc_gates._gradient_weights
            
            # split gradient into hidden and input
            hidden_gradient = combined_input_gradient[:, :self.hidden_size]
            input_gradient[t] = combined_input_gradient[0, self.hidden_size:]
            
            # gradient for previous cell state
            cell_gradient = total_cell_error * self.forget_gates[t:t+1]
        
        # optimizer
        if self.optimizer is not None:
            self.fc_output.weights = self.optimizer.calculate_update(
                self.fc_output.weights, 
                self.gradient_weights_fc_output
            )
            
            self.fc_gates.weights = self.optimizer.calculate_update(
                self.fc_gates.weights, 
                self.gradient_weights_fc_gates
            )
        
        return input_gradient
