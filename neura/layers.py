from abc import ABC, abstractmethod
import numpy as np

class AbstractLayer(ABC):
    @abstractmethod
    def forward():
        pass
    
    @abstractmethod
    def backpropagate():
        pass


class DenseLayer(AbstractLayer):
    '''Dense layer is the simplest layer in neural networks. It passes input through a linear function and then through an element-wise activation function.
    '''
    
    def __init__(self, input_size, output_size, activation_function):
        '''
        Parameters:
            input_size: int
                        size of the input 
            output_size: int
                         size of the output (int)
            activation_function: AbstractActivationFunction
                                 instance of an activation function object
        '''
        self.input_size = input_size
        self.output_size = output_size
        self.activation_function = activation_function #instance
        
        #Xavier Initialization - weigths and biases on interval <-1/n ; 1/n> where n is the input size
        variance = 1/input_size
        self.weights = (2*np.random.rand(input_size, output_size) - 1) * variance
        self.biases = (2*np.random.rand(1,output_size) - 1) * variance
        
    def forward(self,x):
        '''Passes input forward through the layer.

        Parameters:
            x: array_like
               input of the layer
        '''
        self.last_input = x
        output = np.dot(x,self.weights) + self.biases
        return self.activation_function.forward(output)

    def backpropagate(self,x):
        '''Calculates the change vectors for weights and biases of the layer, which the optimzer applies.

        Parameters:
            x: array-like
               derived output vector
        '''
        derived_input = self.activation_function.derived(x)
        self.change_weights = np.dot(self.last_input.T,derived_input)
        self.change_biases = np.sum(derived_input, keepdims = True, axis = 0)
        return np.dot(derived_input,self.weights.T)


