import numpy as np

class LeakyReLU:
    def __init__(self, alpha= 0.01) -> None:
        self.alpha = alpha

    def __call__(self, x):
        return np.where(x > 0, x, x * self.alpha)   
    
    def backward(self):
        raise NotImplementedError