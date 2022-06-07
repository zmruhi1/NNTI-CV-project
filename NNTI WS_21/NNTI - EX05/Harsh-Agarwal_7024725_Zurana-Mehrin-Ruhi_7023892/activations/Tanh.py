import numpy as np

class Tanh:
    def __init__(self) -> None:
        pass

    def __call__(self, x):
        t=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
        return t
    
    def backward(self):
        raise NotImplementedError