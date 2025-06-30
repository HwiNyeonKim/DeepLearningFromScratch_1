import sys
import os

import numpy as np
from matplotlib import pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dataset.mnist import load_mnist
from chapter_3_neural_network import (
    cross_entropy_error,
    sigmoid,
    softmax,
)
from gradient import numerical_gradient


class TwoLayerNet:
    def __init__(
        self, input_size, hidden_size, output_size, weight_init_std=0.01
    ):
        self.params = dict(
            W1=weight_init_std * np.random.randn(input_size, hidden_size),
            b1=np.zeros(hidden_size),
            W2=weight_init_std * np.random.randn(hidden_size, output_size),
            b2=np.zeros(output_size),
        )

    def predict(self, x):
        W1, W2 = self.params["W1"], self.params["W2"]
        b1, b2 = self.params["b1"], self.params["b2"]

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)

        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    def loss(self, x, t):
        """
        Args:
            x (numpy.ndarray): 입력 데이터
            t (numpy.ndarray): 정답 레이블

        Returns:
            float: 손실 함수의 값
        """
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        """가중치 매개변수의 기울기를 구하는 함수
        **너무 느려서 계산 진행이 안 된다. 한 번 루프를 도는데만 수 분 이상 걸리는 듯.

        Args:
            x (numpy.ndarray): 입력 데이터
            t (numpy.ndarray): 정답 레이블

        Returns:
            dict: 가중치 매개변수의 기울기
        """
        def loss_W(W):
            loss = self.loss(x, t)
            return loss

        grads = dict(
            W1=numerical_gradient(loss_W, self.params["W1"]),
            b1=numerical_gradient(loss_W, self.params["b1"]),
            W2=numerical_gradient(loss_W, self.params["W2"]),
            b2=numerical_gradient(loss_W, self.params["b2"]),
        )

        return grads

    def gradient(self, x, t):
        # 순전파
        W1, W2 = self.params["W1"], self.params["W2"]
        b1, b2 = self.params["b1"], self.params["b2"]

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        # 역전파
        dy = (y - t) / batch_num  # softmax + cross-entropy의 미분

        grads = {}
        grads["W2"] = np.dot(z1.T, dy)
        grads["b2"] = np.sum(dy, axis=0)
        dz1 = np.dot(dy, W2.T)
        da1 = dz1 * z1 * (1 - z1)  # sigmoid 미분
        grads["W1"] = np.dot(x.T, da1)
        grads["b1"] = np.sum(da1, axis=0)

        return grads


if __name__ == "__main__":
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    iteration = 10000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.01

    train_loss_list = list()
    train_acc_list = list()
    test_acc_list = list()

    epoch_test_interval = max(train_size / batch_size, 1)

    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    for i in range(iteration):
        # 미니배치 획득
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # 기울기 계산
        # grad = network.numerical_gradient(x_batch, t_batch)
        grad = network.gradient(x_batch, t_batch)

        # 매개변수 갱신
        for key in ("W1", "b1", "W2", "b2"):
            network.params[key] -= learning_rate * grad[key]

        # 학습 결과
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        # 1 epoch마다 테스트 데이터로 평가
        if i % epoch_test_interval == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print(f"epoch: {i / epoch_test_interval}, train acc: {train_acc}, test acc: {test_acc}")

        # print progress with progress bar
        progress = int(i / iteration * 50)  # 50 characters for progress bar
        progress_bar = "[" + "=" * progress + " " * (50 - progress) + "]"
        print(f"{progress_bar} {i / iteration * 100:.2f}%", end="\r")

    plt.plot(range(iteration), train_loss_list)
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.show()
