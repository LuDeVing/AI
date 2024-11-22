import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from Neural_Network import NeuralLayer, ActivationLayer, NeuralNetwork
from matplotlib.widgets import Button


(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


def relu(x):
    return np.maximum(0, x)


def relu_diff(x):
    return np.where(x > 0, 1, 0)


def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)


def cross_entropy_loss(y_pred, y_true):
    samples = y_pred.shape[0]
    y_pred_clipped = np.clip(y_pred, 1e-12, 1. - 1e-12)
    correct_confidences = np.sum(y_true * y_pred_clipped, axis=1)
    loss = -np.mean(np.log(correct_confidences))
    return loss


def cross_entropy_loss_diff(y_pred, y_true):
    return y_pred - y_true


def train():

    nn = NeuralNetwork(loss=cross_entropy_loss, loss_diff=cross_entropy_loss_diff, learning_rate=0.7, iters=2000, display_loss=True)

    nn.add_layer(NeuralLayer(784, 16))  # Input layer to hidden layer
    nn.add_layer(ActivationLayer(relu, relu_diff))
    nn.add_layer(NeuralLayer(16, 16))   # Hidden layer to hidden layer
    nn.add_layer(ActivationLayer(relu, relu_diff))
    nn.add_layer(NeuralLayer(16, 10))    # Hidden layer to output layer
    nn.add_layer(ActivationLayer(softmax, lambda x: 1))  # Output layer with softmax activation

    nn.fit(X_train, y_train)

    nn.save("nn data\\weights and biases.pkl")


train()

nn = NeuralNetwork(loss=cross_entropy_loss, loss_diff=cross_entropy_loss_diff,
                        learning_rate=0.7, iters=1000, display_loss=True)

nn.add_layer(NeuralLayer(784, 16))  # Input layer to hidden layer
nn.add_layer(ActivationLayer(relu, relu_diff))
nn.add_layer(NeuralLayer(16, 16))   # Hidden layer to hidden layer
nn.add_layer(ActivationLayer(relu, relu_diff))
nn.add_layer(NeuralLayer(16, 10))    # Hidden layer to output layer
nn.add_layer(ActivationLayer(softmax, lambda x: 1))  #

nn.load("nn data\\weights and biases.pkl")

y_pred = nn.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

accuracy = np.mean(y_pred_classes == y_true_classes)
print(f"Test accuracy: {accuracy * 100:.2f}%")

print(nn.layers[0].weights)


def visualize_predictions(nn, X_test, y_test):
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.subplots_adjust(bottom=0.25)

    img_display = ax.imshow(np.zeros((28, 28)), cmap='gray', vmin=0, vmax=1)
    title = ax.set_title("")

    # Find all indices of wrong predictions
    y_pred_all = nn.predict(X_test)
    y_pred_classes_all = np.argmax(y_pred_all, axis=1)
    y_true_classes_all = np.argmax(y_test, axis=1)
    wrong_indices = np.where(y_pred_classes_all != y_true_classes_all)[0]

    def show_random_image(event):
        index = np.random.randint(0, len(X_test))
        update_image(index)

    def show_random_wrong_image(event):
        if len(wrong_indices) > 0:
            index = np.random.choice(wrong_indices)
            update_image(index)
        else:
            title.set_text("No wrong predictions found!")
            plt.draw()

    def update_image(index):
        img = X_test[index].reshape(28, 28)
        y_pred = nn.predict(X_test[index].reshape(1, -1))
        predicted_label = np.argmax(y_pred, axis=1)[0]
        true_label = np.argmax(y_test[index])
        img_display.set_data(img)
        img_display.set_clim(0, 1)
        title.set_text(f"True Label: {true_label}, Predicted: {predicted_label}")
        plt.draw()

    ax_button1 = plt.axes([0.1, 0.05, 0.3, 0.1])
    ax_button2 = plt.axes([0.6, 0.05, 0.3, 0.1])

    btn_random_image = Button(ax_button1, 'Random Image')
    btn_random_wrong_image = Button(ax_button2, 'Random Wrong Image')

    btn_random_image.on_clicked(show_random_image)
    btn_random_wrong_image.on_clicked(show_random_wrong_image)

    update_image(0)

    plt.show()


visualize_predictions(nn, X_test, y_test)
