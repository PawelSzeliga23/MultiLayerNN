import numpy as np


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - x ** 2


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x <= 0, 0, 1)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)

def softmax(x):
    exp_x = np.exp(x)
    sum_x = np.sum(exp_x, axis=1, keepdims=True)
    return exp_x / sum_x


def squared_error(y_pred, y_true):
    num_samples = y_pred.shape[0]
    return np.sum((y_true - y_pred) ** 2) / num_samples


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.y_prediction = None
        self.second_layer_output = None
        self.activated_layer_output = None
        self.first_layer_output = None
        self.Weights_1 = np.random.randn(input_size, hidden_size) * 0.01
        self.bias_1 = np.zeros((1, hidden_size))
        self.Weights_2 = np.random.randn(hidden_size, output_size) * 0.01
        self.bias_2 = np.zeros((1, output_size))

    def forward(self, X):
        self.first_layer_output = np.dot(X, self.Weights_1) - self.bias_1
        self.activated_layer_output = relu(self.first_layer_output)
        self.second_layer_output = np.dot(self.activated_layer_output, self.Weights_2) - self.bias_2
        self.y_prediction = softmax(self.second_layer_output)
        return self.y_prediction

    def backward_propagation(self, X, y, learning_rate):
        num_sample = X.shape[0]
        delta_second_layer = y - self.y_prediction
        d_weights_2 = np.dot(self.activated_layer_output.T, delta_second_layer) / num_sample
        d_bias_2 = np.sum(delta_second_layer, axis=0, keepdims=True) / num_sample
        delta_first_layer = np.dot(delta_second_layer, self.Weights_2.T) * relu_derivative(self.first_layer_output)
        d_weights_1 = np.dot(X.T, delta_first_layer) / num_sample
        d_bias_1 = np.sum(delta_first_layer, axis=0, keepdims=True) / num_sample

        self.Weights_2 += learning_rate * d_weights_2
        self.bias_2 += learning_rate * d_bias_2
        self.Weights_1 += learning_rate * d_weights_1
        self.bias_1 += learning_rate * d_bias_1

    def train(self, X, y, learning_rate, epochs, test_x, test_y):
        for epoch in range(epochs + 1):
            y_pred = self.forward(X)
            self.backward_propagation(X, y, learning_rate)
            error = squared_error(y_pred, y)

            if epoch % 10 == 0:
                pre = self.predict(test_x)
                accuracy = np.mean(pre == test_y)
                print(f'Epoch {epoch}, Error : {error:.4f}, Accuracy : {(accuracy * 100):.2f}')

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)
