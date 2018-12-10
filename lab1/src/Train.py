import sys
from datetime import datetime

import numpy as np

import get_data

from NN import NNetwork


def main(argv):
    if len(argv) != 5:
        epochs = 10
        cross_entropy = 0.001
        learn_rate = 0.01
        hidden_size = 500
    else:
        epochs = int(argv[1])
        cross_entropy = float(argv[2])
        learn_rate = float(argv[3])
        hidden_size = int(argv[4])

    N_train = 60000
    N_test = 10000

    train_images = get_data.train_images()[:N_train]
    train_labels = get_data.train_labels()[:N_train]
    test_images = get_data.test_images()[:N_test]
    test_labels = get_data.test_labels()[:N_test]

    X_train = np.zeros((N_train, 784))  # 784 = 28 * 28 from image sizes
    for i, pic in enumerate(train_images):
        X_train[i] = pic.flatten()
    X_test = np.zeros((N_test, 784))
    for i, pic in enumerate(test_images):
        X_test[i] = pic.flatten()
    Y_train = np.zeros((len(train_labels), 10))  # 10 numbers
    for i in range(len(train_labels)):
        Y_train[i][train_labels[i]] = 1
    Y_test = np.zeros((len(test_labels), 10))
    for i in range(len(test_labels)):
        Y_test[i][test_labels[i]] = 1
    network = NNetwork(epochs, cross_entropy, learn_rate, hidden_size)
    network.reset_weights()
    print(str(datetime.now()), 'Initialization successful, training network...')
    network.train(X_train, Y_train)
    print(str(datetime.now()), 'Training ended')
    train_result = network.test(X_train, Y_train)
    print(str(datetime.now()), 'Training data result:', train_result)
    test_result = network.test(X_test, Y_test)
    print(str(datetime.now()), 'Test data precision:', test_result)


if __name__ == "__main__":
    main(sys.argv)
