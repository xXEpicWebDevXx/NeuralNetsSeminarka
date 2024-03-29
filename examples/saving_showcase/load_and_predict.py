from neura.network import Network
import numpy as np

def main(): 
    n = Network.load("saved_model")
    print(n.predict(np.array([[1],[1.5],[100],[0.1]])))



if __name__ == "__main__":
    main()
