from abc import ABC, abstractmethod
from numpy import ones, exp, maximum
from numpy import max as npmax
from numpy import clip


class AbstractActivationFunction():
    '''Abstract class for defining activation functions. All activation functions in neura must inherit from this class.
    '''
    @abstractmethod
    def forward(self, x):
        '''Returns activation functions of all inputs.

        '''
        pass
    @abstractmethod
    def derived(self,x):
        '''Retruns the derivative of last input multiplied with incomming derivatives from layer above/loss funcion.
        '''
        pass
       


class LeakyReLU(AbstractActivationFunction):
    '''Leaky Rectified unit is an activation function that allows for small gradient when unit is not active. Gradient for negative values is set to 0.1 by default.

    Range:
        (-inf,inf)

    Formula:
        LeakyReLU(x) = x if x > 0 else x*gradient_of_negative_input

    '''

    def __init__(self, gradient_of_negative_input = 0.1):
        '''
        Parameters:
            gradient_of_negative_input: gradient of input when input < 0
        '''
        self.gradient_of_negative_input = gradient_of_negative_input
    def forward(self, x):
        self.last_input = x
        return maximum(x * self.gradient_of_negative_input, x)
    def derived(self,x):
        h = ones(shape = self.last_input.shape)
        h[self.last_input <= 0] = self.gradient_of_negative_input
        h *= x
        return h


    



class Sigmoid(AbstractActivationFunction):
    '''Sigmoid is an activation function that outputs values between 0 and 1. Usefull for binary classification.

    Range:
        (0,1)
    
    Formula:
        Sigmoid(x) = 1/(1+exp(-x))
    
        
    Output is clipped within the range of <10^(-10), 1 - 10^(-10)> as to not cause division by zero
    '''
    
    def forward(self,x):
        self.last_output = clip(1 / (1+exp(-x)),10**(-10),1 - 10**(-10))
        return self.last_output
    def derived(self,x):
        return (self.last_output*(1-self.last_output))*x
