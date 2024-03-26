from abc import ABC, abstractmethod
from numpy import sum as npsum
from numpy import log, zeros

class AbstractLossFunction(ABC):
    @abstractmethod
    def forward():
        pass
    @abstractmethod
    def derived():
        pass


class MSE(AbstractLossFunction):
    def forward(self,x,expected_x):
        return npsum((x - expected_x) ** 2) / len(x)
    def derived(self,x, expected_x):
        return 2 * (x - expected_x)


