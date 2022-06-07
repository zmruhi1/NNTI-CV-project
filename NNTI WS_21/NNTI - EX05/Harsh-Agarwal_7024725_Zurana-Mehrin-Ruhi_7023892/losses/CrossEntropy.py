import numpy as np

class CrossEntropy:
    def __init__(self):
        self._eps = 1e-8 # Add eps to CE loss
        
    def __call__(self, Y_pred, Y_true):
        # Assume $Y_true \in {0,1}$ 
        Y_pred = np.clip(Y_pred, self._eps, 1. - self._eps)
        N = Y_pred.shape[0]
        ce = -np.sum(Y_true*np.log(Y_pred+1e-9))/N
        return ce
        
    def backward(self, Y_pred, Y_true):
        raise NotImplementedError