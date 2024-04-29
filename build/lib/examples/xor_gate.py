from neura.layers import DenseLayer
from neura.optimizers import MomentumSGD
from neura.activation_functions import Sigmoid 
from neura.network import Network
from neura.loss_functions import BinaryCrossEntropy
import numpy as np


def main():
    l1 = DenseLayer(2,2,Sigmoid())
    l2 = DenseLayer(2,1,Sigmoid())

    n = Network(BinaryCrossEntropy(), MomentumSGD(0.003,0.9))
    n.add_layer(l1)
    n.add_layer(l2)

    inp = np.array([[0,0],[1,1],[0,1],[1,0]])
    
    out = np.array([[0],[0],[1],[1]])
    n.train(inp,out,100000,2, epoch_info = 100)
    print(n.predict(inp))

if __name__ == "__main__":
    main()                                   
