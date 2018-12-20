import numpy as np
from datetime import datetime


def derivative(func, arg):
    return func(arg) * (1 - func(arg))


def shuffle(a, b):
    p = np.random.permutation(len(a))
    return a[p], b[p]


def logistic_regression(arg):
    return 1 / (1 + np.exp(-arg))


def softmax(arg):
    res = np.zeros(arg.shape)
    s = 0
    for i, row in enumerate(arg):
        res[i] = np.exp(row)
        s += res[i].sum()
    return res / s


class NNetwork:
    weights_hidden = np.array([])
    weights_out = np.array([])
    hidden_layer = np.array([])
    input_layer = np.array([])
    output_layer = np.array([])
    output_layer_expected = np.array([])
    epochs = 100
    cross_entropy_min = 0.05
    learn_rate = 0.01
    hidden_size = 300
    input_size = 28 * 28
    output_size = 10

    def __init__(self, epochs, cross_entropy, learn_rate, hidden_size):
        self.epochs = epochs
        self.cross_entropy_min = cross_entropy
        self.learn_rate = learn_rate
        self.hidden_size = hidden_size
        self.hidden_layer = np.zeros(hidden_size)

    def reset_weights(self):
        self.weights_hidden = 2 * np.random.rand(self.input_size, self.hidden_size) - 1
        self.weights_out = 2 * np.random.rand(self.hidden_size, self.output_size) - 1

    def calc_hidden(self):
        self.hidden_layer = logistic_regression(np.dot(self.input_layer, self.weights_hidden))

    def calc_output(self):
        self.calc_hidden()
        self.output_layer = softmax(np.dot(self.hidden_layer, self.weights_out))

    def correct_weights(self):
        gradient_weights = [
            np.zeros((self.input_size, self.hidden_size)),
            np.zeros((self.hidden_size, self.output_size))
        ]
        delta1 = np.zeros(self.hidden_size)
        delta2 = np.zeros(self.output_size)
        for i in range(self.hidden_size):
            delta2 = self.output_layer - self.output_layer_expected
            gradient_weights[1][i] = np.dot(delta2, self.hidden_layer[i])
        for i in range(self.hidden_size):
            delta1[i] += np.dot(delta2, self.weights_out[i]) * derivative(logistic_regression,
                                                                          self.hidden_layer[i])
        for i in range(self.input_size):
            gradient_weights[0][i] = np.dot(delta1, self.input_layer[i])

        self.weights_hidden -= self.learn_rate * gradient_weights[0]
        self.weights_out -= self.learn_rate * gradient_weights[1]

    def set_input(self, input_layer, label):
        self.input_layer = input_layer
        self.output_layer_expected = label

    def calc_cross_entropy(self, data, labels):
        error = 0.0
        for i in range(len(data)):
            self.set_input(data[i] / 255, labels[i])
            index = self.output_layer_expected.argmax()
            self.calc_output()
            error -= np.log(self.output_layer[index])
        error /= len(data)
        return error

    def train(self, data, labels):
        print(str(datetime.now()), 'Start training...')
        for epoch in range(self.epochs):
            correct = 0
            data, labels = shuffle(data, labels)
            print(str(datetime.now()), 'Shuffled data...')
            for i in range(len(data)):
                self.set_input(data[i] / 255, labels[i])
                self.calc_output()
                if self.output_layer.argmax() == self.output_layer_expected.argmax():
                    correct += 1
                self.correct_weights()
            print(str(datetime.now()), 'Corrected weights...')
            precision = correct / len(data)
            cross_entropy = self.calc_cross_entropy(data, labels)
            print(str(datetime.now()), 'Epoch:', epoch, 'Cross entropy:', cross_entropy, 'Precision:', precision)
            if cross_entropy < self.cross_entropy_min:
                break

    def test(self, data, labels):
        correct = 0
        for i in range(len(data)):
            self.set_input(data[i] / 255, labels[i])
            self.calc_output()
            if self.output_layer.argmax() == self.output_layer_expected.argmax():
                correct += 1
        return correct / len(data)
