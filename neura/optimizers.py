from abc import ABC, abstractmethod


class AbstractOptimizer(ABC):
    '''Abstract class for defining optimizers. All optimizers must inherit from this class
    '''
    @abstractmethod
    def update_parameters(self, *args, **kwargs):
        pass



class SGD(AbstractOptimizer):
    '''Stochastic gradient descend updates the weights and biases by adding the opposite of the change vectors, provided in layer backpropagation, multiplied by learning rate to them.
    
    Formula:
        w = w - learning_rate * dw
        b = b - learning_rate * db
    '''
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update_parameters(self, layers,*args, **kwargs):
        for layer in layers:
            layer.weights -= self.learning_rate * layer.change_weights
            layer.biases -= self.learning_rate * layer.change_biases


class MomentumSGD(AbstractOptimizer):
    '''Stochastic gradient descend with momentum updates weights using momentum. This enables the network to keep momentum from previous weights and biases changes, which can result in faster learning. 
    This method is also less prone to falling into a local minimum. The magnitude of the effect of momentum can be changed by adjusting momentum variable.
    
    Formula:
        w = w - learning_rate*(dw + dw_last * momentum)
        w = b - learning_rate*(db + db_last * momentum)
    '''
    def __init__(self, learning_rate, momentum):
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.changes_dictionary = {}
    def update_parameters(self,layers, *args, **kwargs):
        for layer_index, layer in enumerate(layers):
            if layer_index in self.changes_dictionary:
                last_changes = self.changes_dictionary[layer_index]
                last_weights_changes = last_changes[0]
                last_biases_changes = last_changes[1]
            else:
                last_weights_changes = 0
                last_biases_changes = 0

            weights_change = layer.change_weights + last_weights_changes * self.momentum
            biases_change = layer.change_biases + last_biases_changes * self.momentum



            layer.biases -= biases_change * self.learning_rate
            layer.weights -= weights_change * self.learning_rate
            self.changes_dictionary[layer_index] = [weights_change, biases_change]
