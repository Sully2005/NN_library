# NN_library
A Keras-styel neural netwrok library built from scracth in python. Desgined fotr educational purposes, I have also included files where I implemented the library to solve real problems. The mnist file includes my implementation to solve the digit recognizer task. In the ufcpred file I tested the library out in trying to predict the winner of a UFC fight. 

# Features

-Fully customizable feedforward neural networks

-Layer-wise architecture with activation functions

-Forward and backward propagation implemented manually

-Gradient descent optimization

-Support for common loss functions (MSE, Cross-Entropy)

-Minimal dependencies for easy setup



# Usage

Here is a quick example of how to build and train a simple neural network:

X_train = np.array([[0,0], [0,1],[1,0], [1,1]])
Y_train = np.array([[0], [0], [1], [1]])

network = [
    Dense(2,4), 
    Tanh(), 
    Dense(4,1),
    Sigmoid()
]

train(network, X_train, Y_train, mean_squared_error, mean_squared_error_derivative, 0.01)

        
X_test = np.array([[1,0], [0,1], [1,1], [0,0]])
#Y_test = np.array([[1], [0], [1], [0]])

accuracy = get_Accuracy(X_test, Y_test, network)
print(accuracy)







