class BaseLayer:
    def __init__(self):
        self.trainable = False
        self.weights = None
        self.testing_phase = False

    def forward(self, input_tensor):
        """
        forward pass
        
        args:
            input_tensor: input data to the layer
        """
        raise NotImplementedError

    def backward(self, error_tensor):
        """
        backward pass
        
        args:
            error_tensor: gradient from the next layer
        """
        raise NotImplementedError 