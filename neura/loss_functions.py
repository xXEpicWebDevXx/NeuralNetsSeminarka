from abc import ABC, abstractmethod
import numpy as np


class AbstractLossFunction(ABC):
    @abstractmethod
    def forward():
        pass
    @abstractmethod
    def derived():
        pass


class MSE(AbstractLossFunction):
    def forward(self,x,expected_x):
        return np.sum((x - expected_x) ** 2) / len(x)
    def derived(self,x, expected_x):
        return 2 * (x - expected_x)

class BinaryCrossEntropy(AbstractLossFunction):
    def forward(self, x, expected_x):
        return -expected_x * np.log(x) - (1-expected_x)*np.log(1-x)
    def derived(self, x, expected_x):
        return (-expected_x/x) + (1-expected_x)/(1-x)


