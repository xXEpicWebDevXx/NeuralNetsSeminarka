from neura.layers import DenseLayer
from neura.optimizers import SGD, MomentumSGD
from neura.activation_functions import LeakyReLU, Sigmoid
from neura.network import Network
from neura.loss_functions import MSE, BinaryCrossEntropy
import numpy as np


def main():
    
    l1 = DenseLayer(1,10,Sigmoid())
    l2 = DenseLayer(10,10, Sigmoid())
    l3 = DenseLayer(10,1, Sigmoid())

    n = Network(BinaryCrossEntropy(), MomentumSGD(0.00003, 0.9))
    n.add_layer(l1)
    n.add_layer(l2)
    n.add_layer(l3)



    inp = np.linspace(0,10,10000).astype(np.float64)
    inp = np.reshape(inp,(-1,1))
    out = np.zeros(shape = inp.shape)
    out[inp > 2] = 1
    

    
    n.fit(inp,out,epochs = 500,batch_size = 32) 

    print(n.predict(np.array([[1],[1.5],[5]])))





if __name__ == "__main__":
    main()
