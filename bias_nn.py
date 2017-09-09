####################################################
# Author: Daniel S. Lee
# Date: August 28, 2017
# Title: Time Series Neural Network in Python
####################################################

import numpy as np
import math

# Activation Functions
def ReLu():
    pass

def sigmoid(X, deriv=False):
    if not deriv:
        return 1 / (1 + np.exp(-X))
    else:
        return sigmoid(X) * (1 - sigmoid(X))

def tanh(X, deriv=False):
    if not deriv:
        return (np.exp(X) - np.exp(-X)) / (np.exp(X) + np.exp(-X))
    else:
        return 1 - tanh(X)**2

def linear_activation(X, deriv=False):
    if not deriv:
        return X
    else:
        return -np.ones(X.shape)

def tanh(x):
    return

# Objective Function
def least_squares(y, yhat):
    return np.mean((y - yhat)**2)

def smape(y, yhat):
    return np.mean(2*np.abs(y - yhat)/(np.abs(y) + np.abs(yhat)))

# Neural Network Objects
class Layer:
    """ Stores inputs, outputs and weights in each layer.
    """
    def __init__(self, X=None, y=None, shape=None, isInput=True, isOutput=False):

        # Input layer initialization
        if isInput:
            np.random.seed(0)
            self.S = X
            self.W = np.random.normal(size=shape, scale=1.0E-3)#np.zeros(shape=shape)
            self.Z = None
            self.F = None

        # Hidden layer initialization
        if not isInput and not isOutput:
            np.random.seed(0)
            self.W = np.random.normal(size=shape, scale=1.0E-3) #np.zeros(shape=shape)
            self.S = None
            self.Z = None
            self.F = None
            self.D = None

        # Output layer initialization
        if isOutput:
            self.y = y
            self.S = None
            self.D = None

        self.isInput = isInput
        self.isOutput = isOutput

    # Layer forward propagation
    def forward_propagation(self, activation, add_bias=True):
        S = self.S
        bias = np.ones((S.shape[0], 1))
        if add_bias:
            if self.isInput:
                self.S = np.hstack((S, bias))
                return np.dot(self.S, self.W)              # Output of input layer with the bias node
            elif not self.isInput and not self.isOutput:
                self.Z = activation(S)                     # Activate the input in hidden layer
                self.S = np.hstack((self.Z, bias))         # Add bias to the input in hidden layer
                self.F = activation(S, deriv=True).T       # Calculate the derivative of the input S in the hidden layer. Used later in back prop.
                return np.dot(self.S, self.W)
        else:   # After the first iteration of foward prop., adding bias is unnecessary.
            if self.isInput:
                # print('S input in input node: ', self.S.shape)
                # print('W input in input node: ', self.W.shape)
                return np.dot(self.S, self.W)
            elif not self.isInput and not self.isOutput:
                # print('S input in hidden node: ', self.S.shape)    # Note that activation must occur before bias inclusion
                #                                                    # But this already includes bias. Ignore the last col and activate
                self.Z = activation(S[:,:-1])
                self.S = np.hstack((self.Z, bias))
                self.F = activation(S[:,:-1], deriv=True).T
                return np.dot(self.S, self.W)

class NeuralNetwork:
    """ Performs feedforward, backward prop. and prediction.

        Arguments:
            1) X = 2D-array input matrix
            2) y = 1D-array ouput vector
            3) shape = Neural network architecture. Example: [10,2,3,2]
            4) activation = Activation applied on input matrix

        Methods:
            1) feedforward() ~ Feed forward
            2) backprop() ~ Back propagation
            3) update() ~ Weight update
            3) train()
            4) predict()
    """

    def __init__(self, X, y, activation, dim=[], alpha=0.01, iteration=1000):
        self.layers = []
        self.layer_size = len(dim) - 1  # [506,  13,5,1]
        self.activation = activation
        self.alpha = alpha
        self.y = y
        self.iteration = iteration

        print('Dim after adjustment: ', dim)

        # Generate input layer
        layer = Layer(X=X, y=None, shape=[dim[1]+1,dim[2]], isInput=True, isOutput=False)
        self.layers.append(layer)

        # Generate hidden layer(s)
        for i, d in enumerate(dim[2:-1]):
            i = i + 2
            layer = Layer(X=None, y=None, shape=[dim[i]+1,dim[i+1]], isInput=False, isOutput=False)
            self.layers.append(layer)

        # Generate output layer
        layer = Layer(X=None, y=y, isInput=False, isOutput=True)
        self.layers.append(layer)

    def forward_propagation(self, iteration):
        if iteration == 0: # Run this initially to append the bias column
            for i in range(self.layer_size-1):
                self.layers[i+1].S = self.layers[i].forward_propagation(activation=self.activation, add_bias=True)
                if i == 1:
                    print('Forward Prop Initial Sample: ', self.layers[i+1].S[:2])
        else:
            for i in range(self.layer_size-1):
                # print('S', self.layers[i].S.shape)
                # print('W', self.layers[i].W.shape
                if i == self.layer_size-2:
                    print(True)
                    self.layers[i+1].S = self.layers[i].forward_propagation(activation=self.activation, add_bias=False)
                else:
                    self.layers[i+1].S[:,:-1] = self.layers[i].forward_propagation(activation=self.activation, add_bias=False)

    def back_propagation(self):
        for i in range(self.layer_size - 1, 0, -1): # Enumerate in reverse order
            if i == self.layer_size - 1:
                self.layers[i].D = (self.layers[i].S - self.y).T
            else:
                current_layer = self.layers[i]
                W_nobias = current_layer.W[:-1] # Removes the weight bias
                F_nobias = current_layer.F
                self.layers[i].D = F_nobias * np.dot(W_nobias, self.layers[i+1].D)

    def update_weights(self):
        for i in range(self.layer_size - 2, -1, -1):
            # print('Weights:', self.layers[i].W.shape)
            # print('D:', self.layers[i+1].D.shape)
            # print('Z:', self.layers[i].S.shape)
            if i != 0: # Hidden layer weights update
                self.layers[i].W -= self.alpha * np.dot(self.layers[i+1].D, self.layers[i].S).T
            else: # Input layer weights update
                self.layers[i].W -= self.alpha * np.dot(self.layers[i+1].D, self.layers[i].S).T
            # print("weight shape in updates: ", self.layers[i].W.shape)

    def train(self, objective, criterion, iterations=10000):
        i = 0
        while i < iterations:
            #input(str("Input 't' to continue: "))
            self.forward_propagation(iteration=i)
            i += 1

            print('---------------------------------------------------------------------')
            print('------ Iteration {} -------- for input layer in forward'.format(i))
            print('Input: ')
            print(self.layers[0].S)
            print('Weight:')
            print(self.layers[0].W)

            print('\n ------ Iteration {} -------- for hidden layer in forward'.format(i))
            print('Input w/o bias Activated: ')
            print(self.layers[1].Z[:])
            print('Input w/ bias: ')
            print(self.layers[1].S[:])
            print('Weight: ')
            print(self.layers[1].W[:])
            print('')

            print('\n ------ Iteration {} -------- for Output layer in forward'.format(i))
            print('Prediction: ')
            print(self.layers[2].S)
            print('Output: ')
            print(self.layers[2].y)
            print('')

            self.back_propagation()

            print('---------------------------------------------------------------------')
            print('------ Iteration {} -------- for output layer in backward'.format(i))
            print('D matrix in output layer: ')
            print(self.layers[2].D[0][:10])

            print('\n ------ Iteration {} -------- for hidden layer in backward'.format(i))
            print('D matrix in hidden layer: ')
            print(self.layers[1].D[:2])

            self.update_weights()

            print('---------------------------------------------------------------------')
            print('------ Iteration {} -------- for Hidden layer in update'.format(i))
            print('S matrix used in update: ', self.layers[1].S[:2])
            print(self.layers[1].S.shape)
            print('Update term', self.alpha * np.dot(self.layers[2].D, self.layers[1].S).T)
            print('------ Iteration {} -------- for Input layer in update'.format(i))
            print('S matrix used in update: ', self.layers[0].S[:2])
            print('Update term', self.alpha * np.dot(self.layers[1].D, self.layers[0].S).T)

            last_layer = self.layers[self.layer_size - 1]
            loss = objective(last_layer.y, last_layer.S)
            print(loss)

            # self.update_weights()

            # userinput = input(str("Input 't' to continue: "))
            #
            #
            # # if userinput == 't':
            # #     self.feedforward()
            # #
            # #     """
            # #     print('\n ------ Iteration {} -------- for input layer in forward'.format(i))
            # #     print('Input: ')
            # #     print(self.layers[0].S)
            # #     print('Weight: \n')
            # #     print(self.layers[0].W)
            # #
            # #     print('\n ------ Iteration {} -------- for hidden layer in forward'.format(i))
            # #     print('Input: ')
            # #     print(self.layers[1].S)
            # #     print('Input Activated: ')
            # #     print(self.layers[1].Z)
            # #     print('Weight: ')
            # #     print(self.layers[1].W)
            # #     print('')
            # #
            # #     print('\n ------ Iteration {} -------- for Output layer in output'.format(i))
            # #     print('Output: ')
            # #     print(self.layers[2].S[:15])
            # #     print('')
            # #
            # #     print('Output: ')
            # #     print(self.layers[2].y[:15])
            # #     print('')
            # #     """
            # #
            # #     self.backprop()
            # #
            # #     """
            # #                     print('\n ------ Iteration {} -------- for output layer in backprop'.format(i))
            # #                     print('Output: ')
            # #                     print(self.layers[2].D.shape)
            # #                     print('Weight: ')
            # #     """
            # #
            # #     self.update()
            # #
            # #     """
            # #     print('\n ------ Iteration {} -------- for output layer in update step'.format(i))
            # #     print('Output Recursive Error: ')
            # #     print(self.layers[2].D.shape)
            # #     print('Output Z matrix: ')
            # #     print(self.layers[1].Z)
            # #     print('Current Output Weight: ')
            # #     print(self.layers[1].W)
            # #     print('Output W matrix: ')
            # #     print(np.dot(self.layers[2].D, self.layers[1].Z).T)
            # #     print('Output W matrix with alpha at {].format', self.alpha)
            # #     print(self.alpha * np.dot(self.layers[2].D, self.layers[1].Z).T)
            # #     print('New Output W matrix: ')
            # #     print()
            # #     """
            #
            #     last_layer = self.layers[self.layer_size - 1]
            #     loss = objective(last_layer.y, last_layer.S)
            #
            # i += 1
        #print('W:', self.layers[0].W)
        #print(np.mean(last_layer.y - last_layer.S)**2)

    def predict(self, inputX):
        activation = self.activation
        bias = np.ones((inputX.shape[0], 1))
        inputX = np.hstack((inputX, bias))
        for i, layer in enumerate(self.layers[:-1]):
            if i == 0:
                S = np.dot(inputX, layer.W)             # Create S (Pre-activated Input) Matrix in the first hidden layer
                #print('Input:', inputX[:5])
                #print('W:',layer.W)
                #print('Input*W', S[:5])
                Z = activation(S)                       # Activate the S matrix
                #print("Inside predict", layer.W[:5])
            else:
                S = np.dot(Z, layer.W)                  # Create S matrix - an input for the next layer
                if not self.layers[i+1].isOutput:       # If the next layer is hidden, activate
                    Z = activation(S)
        return S



from sklearn import datasets
import pandas as pd

# diabetes = datasets.load_boston()
# X = diabetes['data'][:5,:5]
# y = diabetes['target'][:5].reshape(X.shape[0], 1)
#
# X = np.apply_along_axis(lambda x: (x - np.mean(x))/np.std(x), 1, X)
# y = np.apply_along_axis(lambda x: (x - np.mean(x))/np.std(x), 0, y)
#
# print(X)
# print(y)
# input('f')

# label = np.repeat(0, 100)
# label[y > 0] = 1
# label = label.reshape(X.shape[0], 1)

# X = np.array([[1,10],
#               [1,2],
#               [1,-2],
#               [2,2]])
# y = np.array([[0],[0.2],[0.4],[0.25]])
#
# X = np.array([[1,2],
#               [1,1]])
# y = np.array([[1],[0]])


# def true_model(X):
#     return 0.01 * X**3 + 0.02 * X ** 3 + 0.01 * X**2 - np.random.normal(0,10,X.shape[0])
#
# X = np.arange(0,30)
# y = true_model(X)
#
# features = X.reshape((30,1))
# target = y.reshape((30,1))
#
# features = np.apply_along_axis(lambda x: (x - np.mean(x))/np.std(x), 0, features)
# target = np.apply_along_axis(lambda x: (x - np.mean(x))/np.std(x), 0, target)
#
# print(features)
# print(target)
# input('f')

def true_model(X):
    return  2 * np.sin(X) - np.random.normal(0,.1,X.shape[0])

X = np.arange(0,100)
y = true_model(X)

features = X.reshape((100,1))
target = y.reshape((100,1))

features = np.apply_along_axis(lambda x: (x - np.mean(x))/np.std(x), 0, features)
target = np.apply_along_axis(lambda x: (x - np.mean(x))/np.std(x), 0, target)

print(features)
print(target)
input('f')

nn = NeuralNetwork(X=features, y=target, dim=[features.shape[0],features.shape[1],5,1], activation=sigmoid, alpha=0.001)
nn.train(objective=least_squares, criterion=1.0, iterations=10000)

#pred = nn.predict(X)
