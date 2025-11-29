import numpy as np

class Sgd:
    def __init__(self, learning_rate: float):
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
        return weight_tensor - self.learning_rate * gradient_tensor 

class SgdWithMomentum:
    def __init__(self, learning_rate, momentum_rate):
        """
        SGD with Momentum optimizer
        """
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v = None  # momentum term

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.v is None:
            self.v = np.zeros_like(weight_tensor)

        # update momentum term
        self.v = self.momentum_rate * self.v - self.learning_rate * gradient_tensor
        
        return weight_tensor + self.v

class Adam:
    def __init__(self, learning_rate, mu, rho):
        """
        Adam optimizer
        
        """
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.v = None  
        self.r = None  
        self.k = 1
        self.eps = np.finfo(float).eps

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.v is None:
            self.v = np.zeros_like(weight_tensor)
            self.r = np.zeros_like(weight_tensor)

        self.v = self.mu * self.v + (1 - self.mu) * gradient_tensor        
        self.r = self.rho * self.r + (1 - self.rho) * gradient_tensor**2
        # bias-corrected
        v_hat = self.v / (1 - self.mu**self.k)
        r_hat = self.r / (1 - self.rho**self.k)

        self.k += 1
        
        return weight_tensor - self.learning_rate * v_hat / (np.sqrt(r_hat) + self.eps) 