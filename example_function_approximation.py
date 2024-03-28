from layers import DenseLayer
from optimizers import SGD, MomentumSGD
from activation_functions import LeakyReLU
from network import Network
from loss_functions import MSE
import numpy as np


def main():
    l1 = DenseLayer(1,50,LeakyReLU())
    l2 = DenseLayer(50,50, LeakyReLU())
    l3 = DenseLayer(50,1,LeakyReLU())

    n = Network(MSE(), MomentumSGD(0.00003, 0.9))
    n.add_layer(l1)
    n.add_layer(l2)
    n.add_layer(l3)

     
    func = lambda x : x ** 2


    inp = np.linspace(0,10,10000).astype(np.float64)
    inp = np.reshape(inp,(-1,1))
    out = func(inp)

    
    n.fit(inp,out,epochs = 500,batch_size = 32) 

    import matplotlib.pyplot as plt
    plt.plot(func(inp))
    plt.plot(n.predict(inp))
    plt.show()






if __name__ == "__main__":
    main()
