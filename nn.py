####################################################
# Author: Daniel S. Lee
# Date: August 18, 2017
# Title: Time Series Neural Network in Python
####################################################

import numpy as np

# Activation Functions
def ReLu():
    pass

def sigmoid(X, deriv=False):
    if not deriv:
        return 1 / (1 + np.exp(-X))
    else:
        return sigmoid(X) * (1 - sigmoid(X))

def linear_activation(X, deriv=False):
    if not deriv:
        return X
    else:
        return -np.ones(X.shape)

def tanh(x):
    return

# Objective Function
def least_squares():
    return

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
            bias = np.ones((X.shape[0], 1))
            self.S = np.hstack((X, bias))
            self.W = np.random.normal(size=shape, scale=1.0E-2)
            self.Z = None
            self.F = None

        # Hidden layer initialization
        if not isInput and not isOutput:
            np.random.seed(0)
            self.W = np.random.normal(size=shape, scale=1.0E-2)
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

    def __init__(self, X, y, activation, dim=[], alpha=0.05):
        self.layers = []
        self.layer_size = len(dim) - 1
        self.activation = activation
        self.alpha = alpha
        self.y = y
        dim[1] = dim[1] + 1 # Add bias

        # Generate input layer
        layer = Layer(X=X, y=None, shape=dim[1:3], isInput=True, isOutput=False)
        self.layers.append(layer)

        # Generate hidden layer(s)
        for i, d in enumerate(dim[2:-1]):
            i = i + 2
            layer = Layer(X=None, y=None, shape=dim[i:i+2], isInput=False, isOutput=False)
            self.layers.append(layer)

        # Generate output layer
        layer = Layer(X=None, y=y, isInput=False, isOutput=True)
        self.layers.append(layer)

    def feedforward(self):
        activation = self.activation
        for i, layer in enumerate(self.layers[:-1]):
            if i == 0:
                self.layers[i+1].S = np.dot(layer.S, layer.W)               # Create S (Pre-activated Input) Matrix in the first hidden layer
                self.layers[i+1].Z = activation(self.layers[i+1].S)         # Activate the S matrix
            else:
                self.layers[i+1].S = np.dot(layer.Z, layer.W)                  # Create S matrix - an input for the next layer
                self.layers[i].F = activation(self.layers[i].S, deriv=True).T  # Differentiate S matrix in the current layer
                if not self.layers[i+1].isOutput:                              # If the next layer is an output, don't activate
                    self.layers[i+1].Z = activation(self.layers[i+1].S)

    def backprop(self):
        for i in range(self.layer_size - 1, 0, -1):                            # Enumerate in reverse order
            if i == self.layer_size - 1:
                self.layers[i].D = (self.layers[i].S - self.y).T
            else:
                self.layers[i].D = self.layers[i].F * np.dot(self.layers[i].W, self.layers[i+1].D)

    def update(self):
        for i in range(self.layer_size - 2, -1, -1):
            if i != 0: # Hidden layer weights update
                self.layers[i].W -= self.alpha * np.dot(self.layers[i+1].D, self.layers[i].Z).T
            else: # Input layer weights update
                self.layers[i].W -= self.alpha * np.dot(self.layers[i+1].D, self.layers[i].S).T

    def train(self, objective, criterion, iterations=100):
        i = 0
        #while True:
        while i < iterations:
            self.feedforward()

            print('\n ------ Iteration {} -------- for input layer in forward'.format(i))
            print('Input: ')
            print(self.layers[0].S)
            print('Weight: \n')
            print(self.layers[0].W)

            print('\n ------ Iteration {} -------- for hidden layer in forward'.format(i))
            print('Input: ')
            print(self.layers[1].S)
            print('Input Activated: ')
            print(self.layers[1].Z)
            print('Weight: ')
            print(self.layers[1].W)
            print('')

            print('\n ------ Iteration {} -------- for Output layer in output'.format(i))
            print(self.layers[2].S[:5])
            print('Output: ')
            print(self.layers[3].S[:15])
            print('')

            print('Output: ')
            print(self.layers[3].y[:15])
            print('')

            self.backprop()
            self.update()
            last_layer = self.layers[self.layer_size - 1]
            loss = objective(last_layer.y, last_layer.S)
            print(loss)
            i += 1
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

diabetes = datasets.load_boston()
X = diabetes['data']
y = diabetes['target'].reshape(X.shape[0], 1)

X = np.apply_along_axis(lambda x: (x - np.mean(x))/np.std(x), 1, X)
y = np.apply_along_axis(lambda x: (x - np.mean(x))/np.std(x), 0, y)

nn = NeuralNetwork(X=X, y=y, dim=[X.shape[0],X.shape[1],13,5,1], activation=sigmoid)
nn.train(objective=smape, criterion=1.0)

pred = nn.predict(X)
