import numpy as np

class Sigmoid:
    def __init__(self) -> None:
        pass

    def __call__(self, x):
        return 1/(1+np.exp(-x))
    
    def backward(self):
        raise NotImplementedError