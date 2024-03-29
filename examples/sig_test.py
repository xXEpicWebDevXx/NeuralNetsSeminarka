from neura.activation_functions import Sigmoid
import numpy as np


s = Sigmoid()
a = np.array([10,100])

print(s.forward(a))

