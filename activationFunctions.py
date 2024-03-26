from abc import ABC, abstractmethod
from numpy import ones, exp, maximum
from numpy import max as npmax

class AbstractActivationFunction():
    @abstractmethod
    def forward():
        pass
    @abstractmethod
    def derived():
        pass
       


class LeakyReLU(AbstractActivationFunction):
    def __init__(self, scale = 0.1):
        self.scale = scale
    def forward(self, x):
        self.last_input = x
        return maximum(x * self.scale, x)
    def derived(self,x):
        h = ones(shape = self.last_input.shape)
        h[self.last_input <= 0] = self.scale
        h *= x
        return h


    




class Sigmoid(AbstractActivationFunction):
    def forward(self,x):
        x = x - npmax(x)
        self.last_output = 1/(1 + exp(-x))
        return 1/(1 + exp(-x))
    def derived(self,x):
        return (self.last_output / (1 - self.last_output)) * x

        
