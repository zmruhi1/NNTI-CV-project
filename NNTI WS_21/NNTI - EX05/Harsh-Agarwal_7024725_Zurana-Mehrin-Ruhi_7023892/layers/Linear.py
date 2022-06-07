import numpy as np
import copy

class Linear:
    def __init__(self, in_features, out_features, bias = True):
        self.weights = np.random.randn(in_features, out_features)
        self.bias    = np.random.randn(1, out_features)        

    def __call__(self, x):
        wx = np.matmul(x, self.weights)
        y = wx + self.bias
        return y

    def backward(self, in_gradient):
        raise NotImplementedError

