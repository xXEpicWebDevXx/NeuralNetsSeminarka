import numpy as np


class Network():
    def __init__(self, loss_function, optimizer):
        self.layers = []
        self.loss_function = loss_function 
        self. optimizer = optimizer
        
    def add_layer(self,layer):
        #TODO check if layer inputs match previous layer
        self.layers.append(layer)
    def __forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    def __backwards(self, y):
        reversed_layers = self.layers[::-1]
        for layer in reversed_layers:
            y = layer.backpropagate(y)
        self.optimizer.update_parameters(self.layers)
    def __shuffle(self,a,b):
        p = np.random.permutation(len(a))

        return a[p],b[p]


    def train(self, x_data, y_data, epochs, batch_size, epoch_info = 10):
        for epoch in range(epochs):
            for batch_index in range(0,x_data.shape[0],batch_size):
                x_batch = x_data[batch_index:batch_index+batch_size]
                y_batch = y_data[batch_index:batch_index+batch_size]
                network_output = self.__forward(x_batch)
                loss = self.loss_function.forward(network_output,y_batch)
                self.__backwards(self.loss_function.derived(network_output,y_batch))
            
            if epoch != 0 and epoch % 10 == 0:
                print(f"[INFO] Epoch {epoch}; Loss : {loss}")

    
    def test(self,x,y):
        network_output = self.__forward(x)
        loss = self.loss_function.forward(network_output, y)
        print(f"[INFO] Test; Loss : {loss}")



    def fit(self, x, y, epochs, batch_size, epoch_info = 10, training_percentage = 0.8, shuffle = True):
        if shuffle:
            x,y = self.__shuffle(x,y)
        training_size = int(len(x)*training_percentage)
        x_training = x[0:training_size]
        y_training = y[0:training_size]
        
        x_testing = x[training_size:]
        y_testing = y[training_size:]

        self.train(x_training,y_training,epochs, batch_size,epoch_info)
        self.test(x_testing,y_testing)


    def predict(self,x):
        return self.__forward(x)
