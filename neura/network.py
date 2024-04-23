import numpy as np
from pickle import load, dump

class Network():
    '''The network class is the implementation of the simplest sequential model of neural network. All layers have on input and one output. No non-linear topology of network is allowed.
    '''

    def __init__(self, loss_function, optimizer):
        '''
        Parameters:
            loss_function: instance of AbstractLossFunction
                           loss function to calculate the loss of the network and the gradient of loss
            optimizer: instance of AbstractOptimizer 
                       Optimizer that adjustst weights and biases based on gradient vectors computed in backpropagation of layers
        '''        
        self.layers = []
        self.loss_function = loss_function 
        self. optimizer = optimizer
        
    def add_layer(self,layer):
        '''Adds new layer to the network
        
        Parameters:
            layer: instance of AbstractLayer
                   layer that will be added to the list of layers
        '''
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
        '''Trains the network using x_data as inputs and y_data as labels.

        Parameters:
            x_data: array_like
                    input data for the network
            y_data: array_like
                    labels for the network
            epochs: int
                    number of epochs for which the model should learn
            batch_size: int
                        size of one batch after which the optimizer will adjust wieghts and biases of layers
            epoch_info: int
                        number of epochs after which the netwrok will print loss of the network
        '''
        for epoch in range(epochs):
            for batch_index in range(0,x_data.shape[0],batch_size):
                x_batch = x_data[batch_index:batch_index+batch_size]
                y_batch = y_data[batch_index:batch_index+batch_size]
                network_output = self.__forward(x_batch)
                loss = self.loss_function.forward(network_output,y_batch)
                self.__backwards(self.loss_function.derived(network_output,y_batch))
            
            if epoch != 0 and epoch % epoch_info == 0:
                print(f"[INFO] Epoch {epoch}; Loss : {loss}")

    
    def test(self,x,y):
        '''Tests the network and prints the loss of it.

        Parameters:
            x: array_like
               input data for the network
            y: array_like
               labels for the network
        
        Returns:
            float: loss of the network
        '''
        network_output = self.__forward(x)
        loss = self.loss_function.forward(network_output, y)
        print(f"[INFO] Test; Loss : {loss}")
        return loss


    def fit(self, x, y, epochs, batch_size, epoch_info = 10, training_percentage = 0.8, shuffle = True):
        '''Splits the data into testing data and training data, possibly shuffles them and trains the network. After that it tests the network.
        
        Parameters:
            x_data: array_like
                    input data for the network
            y_data: array_like
                    labels for the network
            epochs: int
                    number of epochs for which the model should learn
            batch_size: int
                        size of one batch after which the optimizer will adjust wieghts and biases of layers
            epoch_info: int
                        number of epochs after which the netwrok will print loss of the network
            training_percentage: float in range (0,1)
                                 the percentage of data that will be used for training the network
            shuffle: bool
                     whether or not will the data be shuffle before training
        
        Returns:
            float: loss of the network
        '''
        
        if shuffle:
            x,y = self.__shuffle(x,y)
        training_size = int(len(x)*training_percentage)
        x_training = x[0:training_size]
        y_training = y[0:training_size]
        
        x_testing = x[training_size:]
        y_testing = y[training_size:]

        self.train(x_training,y_training,epochs, batch_size,epoch_info)
        return self.test(x_testing,y_testing)
    
    def predict(self,x):
        '''Predicts output for input x
        
        Parameters:
            x: array_like
                    input for the network

        Returns:
            array_like: the output predicted by the network
        '''
        return self.__forward(x)

    def __correct_filename(filename):
        if '.' in filename:
            filename = filename.split('.')[0]
        filename += ".pkl"
        return filename

    def save(self, filename):
        '''Saves network into a new .pkl file.

        Parameters:
            filename: string
                      name of the file into which the model should be saved; can be with or without the .pkl file extension

        '''
        filename = Network.__correct_filename(filename)
        with open(filename,"wb") as file:
            dump(self,file)

    def load(filename):
        '''Loads a network from a file. Called from network class, not its instance.

        Parameters:
            filename: string
                      name of the file from which the model should be loaded; can be wiht or without the .pkl file extension
        
        Returns:
            Network: new Network instance with all it's properties
                      
        '''
        filename = Network.__correct_filename(filename)
        with open(filename,"rb") as file:
            return load(file)
