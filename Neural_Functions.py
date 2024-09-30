import numpy as np


class NeuralFunctions:

    @staticmethod
    def linear_loss(y_predicted, y):
        return np.mean(np.power(y - y_predicted, 2))

    @staticmethod
    def linear_loss_diff(y_predicted, y):
        return 2 / len(y) * (y_predicted - y)

    @staticmethod
    def linear_activation(x):
        return x

    @staticmethod
    def linear_activation_diff(x):
        return np.ones_like(x)

    @staticmethod
    def sigmoid_activation(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_activation_diff(x):
        sigmoid = NeuralFunctions.sigmoid_activation(x)
        return sigmoid * (1 - sigmoid)

    @staticmethod
    def relu_activation(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_activation_diff(x):
        return np.where(x > 0, 1, 0)

    @staticmethod
    def tanh_activation(x):
        return np.tanh(x)

    @staticmethod
    def tanh_activation_diff(x):
        return 1 - np.power(np.tanh(x), 2)
