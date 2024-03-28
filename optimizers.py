from abc import ABC, abstractmethod


class AbstractOptimizer(ABC):
    @abstractmethod
    def update_parameters():
        pass



class SGD(AbstractOptimizer):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update_parameters(self, layers,*args, **kwargs):
        for layer in layers:
            layer.weights -= self.learning_rate * layer.change_weights
            layer.biases -= self.learning_rate * layer.change_biases


class MomentumSGD(AbstractOptimizer):
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
