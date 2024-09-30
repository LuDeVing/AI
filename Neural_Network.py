import os
import pickle

import numpy as np


class NeuralLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(output_size) - 0.5
        self.input = None
        self.output = None

    def forward_propagation(self, X):
        self.input = X
        self.output = np.dot(X, self.weights) + self.bias
        return self.output

    def backward_propagation(self, error, lr):
        dw = np.dot(self.input.T, error)
        db = np.sum(error, axis=0)

        dx = np.dot(error, self.weights.T)

        self.weights -= lr * dw
        self.bias -= lr * db

        return dx


class ActivationLayer:
    def __init__(self, activation, activation_diff):
        self.activation = activation
        self.activation_diff = activation_diff
        self.input = None
        self.output = None

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward_propagation(self, error, lr):
        return self.activation_diff(self.input) * error


class NeuralNetwork:
    def __init__(self, loss, loss_diff, learning_rate=0.01, iters=1000, display_loss=False):
        self.layers = []
        self.loss = loss
        self.loss_diff = loss_diff
        self.lr = learning_rate
        self.iters = iters
        self.display_loss = display_loss

    def add_layer(self, layer):
        self.layers.append(layer)

    def fit(self, X, y):
        samples = X.shape[0]

        for _ in range(self.iters):
            print(_)
            y_predicted = X
            for layer in self.layers:
                y_predicted = layer.forward_propagation(y_predicted)

            if self.display_loss:
                loss = self.loss(y_predicted, y)
                print(f"Loss: {loss}")

            error = self.loss_diff(y_predicted, y) / samples
            for layer in reversed(self.layers):
                error = layer.backward_propagation(error, self.lr)

    def predict(self, X):
        y_predicted = X
        for layer in self.layers:
            y_predicted = layer.forward_propagation(y_predicted)
        return y_predicted

    def save(self, filename):
        weights_biases = []
        for layer in self.layers:
            if isinstance(layer, NeuralLayer):
                weights_biases.append((layer.weights, layer.bias))
        with open(filename, 'wb') as file:
            pickle.dump(weights_biases, file)

    def load(self, filename):
        with open(filename, 'rb') as file:
            weights_biases = pickle.load(file)

        weights_biases_idx = 0
        for layer in self.layers:
            if isinstance(layer, NeuralLayer):
                weights, biases = weights_biases[weights_biases_idx]
                layer.weights = weights
                layer.bias = biases
                weights_biases_idx += 1
