import numpy as np


class NeuralLayer:
    def __init__(self, layer_size, previous_features):
        self.layer_size = layer_size
        self.weights = np.random.randn(previous_features, self.layer_size) * 0.01
        self.bias = np.zeros((1, self.layer_size))
        self.values = np.zeros((1, layer_size))

    def forward(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        self.values = 1 / (1 + np.exp(-linear_output))  # Sigmoid activation
        return self.values

    def backward(self, X, dvalues, learning_rate):
        m = X.shape[0]

        dvalues *= self.values * (1 - self.values)

        dw = np.dot(X.T, dvalues) / m
        db = np.sum(dvalues, axis=0, keepdims=True) / m

        dinput = np.dot(dvalues, self.weights.T)

        self.weights -= learning_rate * dw
        self.bias -= learning_rate * db

        return dinput


class NeuralNetwork:
    def __init__(self, n_layers, layer_size, lr=0.01, iters=1000):
        self.n_layers = n_layers
        self.layer_size = layer_size
        self.lr = lr
        self.iters = iters
        self.layers = []
        self.classes = None

    def fit(self, X, y):
        samples, features = X.shape
        self.classes = np.unique(y)
        num_classes = len(self.classes)

        self.layers.append(NeuralLayer(self.layer_size, features))
        for _ in range(self.n_layers - 1):
            self.layers.append(NeuralLayer(self.layer_size, self.layer_size))
        self.layers.append(NeuralLayer(num_classes, self.layer_size))

        for _ in range(self.iters):
            # Forward pass
            input_data = X
            for layer in self.layers:
                input_data = layer.forward(input_data)

            # Compute the loss (using cross-entropy)
            logits = self.layers[-1].values
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            y_one_hot = np.eye(num_classes)[y]
            loss = -np.mean(np.sum(y_one_hot * np.log(probabilities + 1e-15), axis=1))

            # Backward pass
            dvalues = probabilities - y_one_hot
            for layer in reversed(self.layers):
                dvalues = layer.backward(
                    self.layers[self.layers.index(layer) - 1].values if self.layers.index(layer) > 0 else X, dvalues,
                    self.lr)

    def predict(self, X):
        input_data = X
        for layer in self.layers:
            input_data = layer.forward(input_data)
        return np.argmax(input_data, axis=1)

    def evaluate(self, X, y):
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy



