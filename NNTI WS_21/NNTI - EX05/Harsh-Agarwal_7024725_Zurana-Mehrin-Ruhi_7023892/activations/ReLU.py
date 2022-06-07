import numpy as np

class ReLU:
    def __init__(self):
        pass
    
    def __call__(self, x):
        a = np.maximum(0,x)
        return a


    def backward(self, in_gradient):
        raise NotImplementedError