import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from chapter_3_neural_network import cross_entropy_error, softmax
from gradient import numerical_gradient


class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss


if __name__ == "__main__":
    net = SimpleNet()
    print(f"net.W: {net.W}")

    x = np.array([0.6, 0.9])
    p = net.predict(x)
    print(f"p: {p}")
    print(f"np.argmax(p): {np.argmax(p)}")  # 최댓값의 인덱스

    t = np.array([0, 0, 1])  # 정답 레이블
    print(f"net.loss(x, t): {net.loss(x, t)}")

    f = lambda W: net.loss(x, t)
    dW = numerical_gradient(f, net.W)
    print(f"dW: {dW}")
