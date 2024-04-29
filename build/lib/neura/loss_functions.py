from abc import ABC, abstractmethod
import numpy as np


class AbstractLossFunction(ABC):
    '''Abstract class for defining loss functions. All loss functions in neura must inherit from this class.
    '''
    @abstractmethod
    def forward(self,x, label):
        '''Calculates the loss of the network.

        Parameters:
            x: array_like
               network output
            label: array_like
                   label that was expected
        '''
        pass
    @abstractmethod
    def derived(self,x,label):
        '''Calculates the gradient of the loss functions based on label and input.
        
        Parameters:
            x: array_like
               network output
            label: array_like
                   label that was expected
        '''
        pass


class MSE(AbstractLossFunction):
    '''Mean square error is a loss functions that caculates the square of the difference between the input and the label. Useful for approximation of mathematical functions.

    Range of expected input:
        (-inf, inf)
    
    Formula:
        MSE(x, label) = (x - label) ^ 2
    '''

    def forward(self,x,label):
        return np.sum((x - label) ** 2) / len(x)
    def derived(self,x, label):
        return 2 * (x - label)

class BinaryCrossEntropy(AbstractLossFunction):
    '''Binary cross entropy computes the cross-entropy between the input and the expected input. Useful for binary classification.
    
    Range of expected input:
        {0;1}
    
    Formula:
        BinaryCrossEntropy(x,label) = -label * ln(x) - (1-label) * ln(1-x)
    '''


    def forward(self, x, label):
        x = np.clip(x,10**(-7),1 - 10**(-7))
        return np.sum(-label * np.log(x) - (1-label)*np.log(1-x)) / (len(x)*2)
    def derived(self, x, label):
        return ((-label/x) + (1-label)/(1-x))



