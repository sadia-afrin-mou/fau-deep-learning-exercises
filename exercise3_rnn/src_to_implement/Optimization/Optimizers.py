import numpy as np

class Optimizer:
    def __init__(self):
        self.regularizer = None
    
    def add_regularizer(self, regularizer):
        """
        add regularizer to optimizer
        
        args:
            regularizer: regularizer instance
        """
        self.regularizer = regularizer

class Sgd(Optimizer):
    def __init__(self, learning_rate: float):
        super().__init__()
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        """
        a basic gradient descent update

        args:
            weight_tensor: current weights
            gradient_tensor: gradient of the loss w.r.t the weights
            
        returns:
            updated weight_tensor
        """
        original_weights = weight_tensor.copy()
        
        weight_tensor = weight_tensor - self.learning_rate * gradient_tensor

        # regularization
        if self.regularizer is not None:
            weight_tensor = weight_tensor - self.learning_rate * self.regularizer.calculate_gradient(original_weights)
        
        return weight_tensor

class SgdWithMomentum(Optimizer):
    def __init__(self, learning_rate, momentum_rate):
        """
        SGD with Momentum optimizer
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v = None  # momentum term

    def calculate_update(self, weight_tensor, gradient_tensor):
        original_weights = weight_tensor.copy()
        
        if self.v is None:
            self.v = np.zeros_like(weight_tensor)

        # update momentum term and then weight update
        self.v = self.learning_rate * gradient_tensor + self.momentum_rate * self.v
        weight_tensor = weight_tensor - self.v
        
        # regularization
        if self.regularizer is not None:
            weight_tensor = weight_tensor - self.learning_rate * self.regularizer.calculate_gradient(original_weights)

        return weight_tensor

class Adam(Optimizer):
    def __init__(self, learning_rate, mu, rho):
        """
        Adam optimizer
        
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.v = None  
        self.r = None  
        self.k = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        original_weights = weight_tensor.copy()
        
        if self.v is None:
            self.v = np.zeros_like(weight_tensor)
            self.r = np.zeros_like(weight_tensor)

        self.k += 1
        
        self.v = self.mu * self.v + (1 - self.mu) * gradient_tensor        
        self.r = self.rho * self.r + (1 - self.rho) * gradient_tensor**2
        
        # bias-corrected
        v_hat = self.v / (1 - self.mu**self.k)
        r_hat = self.r / (1 - self.rho**self.k)

        weight_tensor = weight_tensor - self.learning_rate * v_hat / (np.sqrt(r_hat) + np.finfo(float).eps)
        
        # regularization
        if self.regularizer is not None:
            weight_tensor = weight_tensor - self.learning_rate * self.regularizer.calculate_gradient(original_weights)
        
        return weight_tensor