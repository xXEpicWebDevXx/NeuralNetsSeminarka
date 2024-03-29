from abc import ABC, abstractmethod
from numpy import ones, exp, maximum
from numpy import max as npmax
from numpy import clip


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


    




class Softmax(AbstractActivationFunction):
    def forward(self,x):
        pass    
    def derived(self,x):
        self.last_output = clip(self.last_output,10**(-7),1 - 10**(-7))
        return (self.last_output / (1 - self.last_output)) * x

class Sigmoid(AbstractActivationFunction):
    def forward(self,x):
        self.last_output = clip(1 / (1+exp(-x)),10**(-10),1 - 10**(-10))
        return self.last_output
    def derived(self,x):
        return (self.last_output*(1-self.last_output))*x
