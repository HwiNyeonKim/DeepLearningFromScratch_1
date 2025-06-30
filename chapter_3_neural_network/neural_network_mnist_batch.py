import os
import pickle
import sys

import numpy as np
from sigmoid_function import sigmoid
from softmax_function import softmax

from dataset.mnist import load_mnist

# 경로를 먼저 추가
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(
        normalize=True, one_hot_label=True
    )

    return x_train, t_train, x_test, t_test


def init_network():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    weight_path = os.path.join(base_dir, "../dataset/sample_weight.pkl")
    weight_path = os.path.abspath(weight_path)  # 절대경로로 변환
    with open(weight_path, "rb") as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)

    a2 = np.dot(z1, W2) + b2

    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3

    y = softmax(a3)

    return y


if __name__ == "__main__":
    _, _, x_test, t_test = get_data()
    network = init_network()

    batch_size = 100
    accuracy_cnt = 0
    for i in range(0, len(x_test), batch_size):
        x_batch = x_test[i : i + batch_size]
        y_batch = predict(network, x_batch)
        p = np.argmax(y_batch, axis=1)
        accuracy_cnt += np.sum(p == np.argmax(t_test[i : i + batch_size], axis=1))

    print("Accuracy:" + str(float(accuracy_cnt) / len(x_test)))
