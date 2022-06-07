import activations
import layers

class Model:
    def __init__(self, components) -> None:
        '''
        expects a list of components of the model in order with which they must be applied
        '''
        self.components = components

    def forward(self, x):
        '''
        performs forward pass on the input x using all components from self.components
        '''
        out = x
        for comp in self.components:
          out = comp(out)
        return(out)
        
    def backward(self, in_grad):
        '''
        expects in_grad - a gradient of the loss w.r.t. output of the model
        in_grad must be of the same size as the output of the model

        returns dictionary, where 
            key - index of the component in the component list
            value - value of the gradient for that component
        '''
        grads = {}
        grads['activation2'] = self.components[3].grad(in_grad)
        grads['layer2']      = self.components[2].grad(grads['activation2'])
        grads['activation1'] = self.components[1].grad(grads['layer2'][1])
        grads['layer1']      = self.components[0].grad(grads['activation1'])
        return grads
      
    def update_parameters(self, grads, lr): 
        '''
        performs one gradient step with learning rate lr for all components
        '''
        self.components[2].weights = self.components[2].weights - 0.001*grads['layer2'][0]
        self.components[0].weights = self.components[0].weights - 0.001*grads['layer1'][0]
