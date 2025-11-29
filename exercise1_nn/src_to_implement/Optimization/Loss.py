import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        self._prediction_tensor = None
        
    def forward(self, prediction_tensor, label_tensor):
        """
        cross entropy loss func
        
        args:
            prediction_tensor: network predictions
            label_tensor: true labels
            
        returns:
            loss value
        """
        self._prediction_tensor = prediction_tensor
        
        epsilon = np.finfo(float).eps
        prediction_tensor = np.clip(prediction_tensor, epsilon, 1 - epsilon)
        
        return -np.sum(label_tensor * np.log(prediction_tensor))
        
    def backward(self, label_tensor):
        """
        gradient of cross entropy loss func
        
        args:
            label_tensor: true labels
            
        Returns:
            error tensor for previous layer
        """
        epsilon = np.finfo(float).eps
        prediction_tensor = np.clip(self._prediction_tensor, epsilon, 1 - epsilon)
        
        return -label_tensor / prediction_tensor 