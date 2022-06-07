import activations
import layers
import numpy as np
class Model:
    def __init__(self, components) -> None:
        '''
        expects a list of components of the model in order with which they must be applied
        '''
        self.components = components
        self.velocity = None
        self.cache = None
        self.grads_first_moment = None
        self.grads_second_moment = None

    def forward(self, x):
        '''
        performs forward pass on the input x using all components from self.components
        '''
        for component in self.components:
            x = component(x)
        return x
        
    def backward(self, in_grad):
        '''
        expects in_grad - a gradient of the loss w.r.t. output of the model
        in_grad must be of the same size as the output of the model

        returns dictionary, where 
            key - index of the component in the component list
            value - value of the gradient for that component
        '''
        num_components = len(self.components)
        grads = {}
        for i in range(num_components-1, -1, -1):
            component = self.components[i]
            if component.get_type() == 'activation':
                in_grad = component.grad(in_grad)
            elif component.get_type() == 'layer':
                weights_grad, in_grad = component.grad(in_grad)
                grads[i] = weights_grad
            else:
                raise Exception
        return grads

    def update_parameters(self, grads, lr):
        '''
        performs one gradient step with learning rate lr for all components
        ''' 
        for i, grad in grads.items():
            assert self.components[i].get_type() == 'layer'
            self.components[i].weights = self.components[i].weights - lr * grad

    def sgd_momentum(self, grads, lr, momentum):
        # your implementation of SGD with momentum goes here
        if self.velocity == None:
            self.velocity = {i: np.zeros_like(grad) for i, grad in grads.items()}
        
        for i, grad in grads.items():
            assert self.components[i].get_type() == 'layer'
            self.velocity[i] = momentum * self.velocity[i] - lr * grad
            self.components[i].weights = self.components[i].weights + self.velocity[i]
  
    def ada_grad(self, grads, lr):
        # your implementation of AdaGrad goes here
        if self.cache == None:
            self.cache = {i: np.zeros_like(grad) for i, grad in grads.items()}
        
        for i, grad in grads.items():
            assert self.components[i].get_type() == 'layer'
            self.cache[i] = self.cache[i] + grad**2
            self.components[i].weights = self.components[i].weights - (lr*grad)/(np.sqrt(self.cache[i]) + np.finfo(np.float32).eps)

    def adam(self, grads, lr, t, beta):
        # your implementation of Adam goes here
        beta1, beta2 = beta
        if self.grads_first_moment == None and self.grads_second_moment == None:
            self.grads_first_moment = {i: np.zeros_like(grad) for i, grad in grads.items()}
            self.grads_second_moment = {i: np.zeros_like(grad) for i, grad in grads.items()}

        for i, grad in grads.items():
            assert self.components[i].get_type() == 'layer'
            self.grads_first_moment[i] = beta1 * self.grads_first_moment[i] + (1. - beta1) * grad
            self.grads_second_moment[i] = beta2 * self.grads_second_moment[i] + (1. - beta2) * grad**2

            m_k_hat = self.grads_first_moment[i] / (1. - beta1**(t))
            r_k_hat = self.grads_second_moment[i] / (1. - beta2**(t))

            self.components[i].weights = self.components[i].weights - lr * m_k_hat / (np.sqrt(r_k_hat) + np.finfo(np.float32).eps)

