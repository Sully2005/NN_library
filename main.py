import numpy as np
import pandas as pd


#Something to note when using this library is that when initiliasing a network, you do it in the form of 
# network = [
#     Dense(features, no_of_neurons), 
#     activationFunction, 
#     Dense(no_of_neurons, outputsDesired)
# ]
#Also please ensure that Y_train or Y_test are of the form (batch_size, 1) and not (batch_size,) as this causes 
#problems in backpropogation

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward_pass(self, input):
        pass

    def backprop(self, output_derivative, learning_rate):
        pass


class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.biases = np.zeros((1, output_size))
        

    def forward_pass(self,input):
        self.input = input
        return np.dot(self.input, self.weights) + self.biases
    
    def backprop(self, output_derivative, learning_rate):
        weight_gradient = np.dot( self.input.T, output_derivative)
        input_gradient = np.dot(output_derivative, self.weights.T)
        bias_gradient = np.sum(output_derivative, axis=0, keepdims=True)

        self.biases -= learning_rate * bias_gradient
        self.weights -= learning_rate * weight_gradient

        #Use this as to continue backprop in the case of additional layers
        return input_gradient
    

class Sigmoid(Layer):

    def forward_pass(self, input):
        self.input = input
        self.output  = 1/(1+ np.exp(-self.input))
        return self.output
        
    def backprop(self, output_derivative):
        sigmoid_derivative = self.output *( 1 - self.output)

        return np.multiply(sigmoid_derivative, output_derivative)
    

class Relu(Layer):

    def forward_pass(self, input):
        self.input = input
        self.output = np.maximum(0, self.input)
        return self.output
    
    def backprop(self, output_derivative):
        relu_derivative = self.input > 0
        return np.multiply(output_derivative, relu_derivative)
    

class Tanh(Layer):

    def forward_pass(self, input):
        self.input = input
        self.output = np.tanh(self.input)
        return self.output
    
    def backprop(self, output_derivative):
        tanh_derivative  = 1 -  self.output** 2
        return np.multiply(output_derivative, tanh_derivative)
    

class Softmax(Layer):

    def forward_pass(self, input):
        shifted_input = input - np.max(input, axis=1, keepdims=True)
        exp_shifted = np.exp(shifted_input)
        self.output = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)

        return self.output
    
    def backprop(self,output_derivative):
        #simplifies nicely
        return output_derivative
    
     
    

def split_training_data(X, Y, percentage_testing):
    if percentage_testing > 1 or percentage_testing < 0:
        raise ValueError("Percentage must be between 0 and 1")
    rows = X.shape[0]

    rows_testing = int(rows * percentage_testing)

    indices = np.random.permutation(rows)
    training_indices = indices[: rows_testing]
    testing_indices = indices[rows_testing:]

    X_test = X[testing_indices]
    Y_test =  Y[testing_indices]

    X_train = X[training_indices]
    Y_train = Y[training_indices]


    return X_train, Y_train, X_test, Y_test

def mean_squared_error(outputs, answers):

    #error for the entire batch 
    return (np.mean((outputs - answers) ** 2))


    

def mean_squared_error_derivative(outputs, answers):
    return (2 * (outputs - answers)) / outputs.shape[0]


def cross_entropy(outputs, answers):
    #prevent log(0)
    min = 0.0000001
    outputs = np.clip(outputs,min, 1- min  )
    return -np.mean(np.sum(answers * np.log(outputs), axis=1))

def cross_entropy_derivative(outputs, answers):
    return (outputs - answers) / outputs.shape[0]



    


def predict(network, input):
    output = input
    for layer in network: 
        output = layer.forward_pass(output)

    return output

def one_hot_encoding(Y):
    one_hot = np.zeros((Y.size, Y.max() + 1))
    one_hot[np.arange(Y.size), Y] = 1 

    return one_hot



 
def train(network, X_train, Y_train, loss, loss_derivative, learning_rate, iterations=1500):

    for iteration in range(iterations):
        error = 0
        
        outputs = predict(network, X_train)
        

        error = loss(outputs, Y_train)
        output_gradient = loss_derivative(outputs, Y_train)

        for layer in reversed(network):
            if isinstance(layer, Dense):
                output_gradient = layer.backprop(output_gradient, learning_rate)
            else:
                output_gradient = layer.backprop(output_gradient)


        if (iteration % 100 == 0):
            print(f"Iteration: {iteration},   Error: {error}")


def get_Accuracy(X_test, Y_test, network):
    predictions = predict(network, X_test)
    
    if Y_test.ndim == 1:
        predictions = (predictions > 0.5).astype(int).flatten()
        targets = Y_test

    else: 
        predictions = np.argmax(predictions, axis = 1)
        targets = np.argmax(Y_test, axis = 1)

    correct = np.sum(predictions == targets)

    
    return (correct / len(Y_test)) * 100


            




    
                


