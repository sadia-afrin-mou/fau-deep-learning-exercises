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