from .layers import DenseLayer
from .optimizers import SDG
from .activationFunctions import LeakyReLU
from .network import Network
from .lossFunctions import MSE
import numpy as np


def main():
    l1 = DenseLayer(2,2,LeakyReLU())
    l2 = DenseLayer(2,1,LeakyReLU())

    n = Network(MSE(), SDG(0.003))
    n.add_layer(l1)
    n.add_layer(l2)

    inp = np.array([[0,0],[1,1],[0,1],[1,0]])
    
    out = np.array([[0],[0],[1],[1]])
    n.train(inp,out,10000,2)
    print(n.predict(inp))

if __name__ == "__main__":
    main()                                   
