import numpy as np
import copy

class Dropout:
    def __init__(self, layer, p : float = 0.5):
        self.p = p
        self.layer = layer
        self.weights = self.layer.weights
        self.bias = self.layer.bias

    def __call__(self, x):
        '''
            apply inverted dropout. Store the mask that you generate with probability p in
            self.mask
        '''
        keep_prob = 1 - self.p
        h = np.random.random(x.shape)
        self.mask = (h < keep_prob) / keep_prob
        return self.layer(self.mask * x)

    def get_type(self):
        return 'layer'

    def grad(self, in_gradient):
        '''
            Apply the mask to the backward pass of a linear layer.
            The return values () are similar to Linear.py
        '''
        return self.layer.grad(self.mask * in_gradient)
