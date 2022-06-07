import numpy as np

class MSELoss:
    def __init__(self) -> None:
        pass

    def __call__(self, y_true, y_pred):
        return np.square(np.subtract(y_true,y_pred)).mean()
    
    def backward(self):
        raise NotImplementedError