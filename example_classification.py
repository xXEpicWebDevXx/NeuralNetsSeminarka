from layers import DenseLayer
from optimizers import SGD, MomentumSGD
from activationFunctions import Sigmoid
from network import Network
from lossFunctions import BinaryCrossEntropy, MSE
import numpy as np


def main():
    l1 = DenseLayer(1,50,Sigmoid())
    l2 = DenseLayer(50,50, Sigmoid())
    l3 = DenseLayer(50,1,Sigmoid())

    n = Network(MSE(), MomentumSGD(0.003, 0.9))
    n.add_layer(l1)
    n.add_layer(l2)
    n.add_layer(l3)

    inp = np.linspace(0,10,1000).astype(np.float64)
    inp = np.reshape(inp,(-1,1))

    out = np.zeros(shape = (inp.shape[0],1))
    out[inp > 3] = 1

    print(out)

    n.fit(inp,out,epochs = 100,batch_size = 32)


    print(n.predict(inp))





if __name__ == "__main__":
    main()
