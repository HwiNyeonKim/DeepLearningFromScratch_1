import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from affine_layer import Affine
from relu_layer import Relu
from softmax_with_loss import SoftmaxWithLoss
from chapter_4_neural_network_training import numerical_gradient


class TwoLayerNet:
    def __init__(
        self, input_size, hidden_size, output_size, weight_init_std=0.01
    ):
        self.params = dict()  # dict is OrderedDict from Python 3.7

        # Initialize weights and biases
        self.params["W1"] = weight_init_std * np.random.randn(
            input_size, hidden_size
        )
        self.params["b1"] = np.zeros(hidden_size)

        self.params["W2"] = weight_init_std * np.random.randn(
            hidden_size, output_size
        )
        self.params["b2"] = np.zeros(output_size)

        # Create layers
        self.layers = dict()
        self.layers["Affine1"] = Affine(self.params["W1"], self.params["b1"])
        self.layers["Relu1"] = Relu()
        self.layers["Affine2"] = Affine(self.params["W2"], self.params["b2"])

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        """
        Args:
            x (numpy.ndarray): 입력 데이터
            t (numpy.ndarray): 정답 데이터

        Returns:
            float: 손실 함수의 값
        """
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)

        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        """
        Args:
            x (numpy.ndarray): 입력 데이터
            t (numpy.ndarray): 정답 데이터

        Returns:
            dict: 각 가중치와 편향의 기울기
        """
        loss_W = lambda W: self.loss(x, t)

        grads = dict()
        grads["W1"] = numerical_gradient(loss_W, self.params["W1"])
        grads["b1"] = numerical_gradient(loss_W, self.params["b1"])
        grads["W2"] = numerical_gradient(loss_W, self.params["W2"])
        grads["b2"] = numerical_gradient(loss_W, self.params["b2"])

        return grads

    def gradient(self, x, t):
        """
        Args:
            x (numpy.ndarray): 입력 데이터
            t (numpy.ndarray): 정답 데이터

        Returns:
            dict: 각 가중치와 편향의 기울기
        """
        # forward propagation
        self.loss(x, t)

        # backward propagation
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = dict()
        grads["W1"] = self.layers["Affine1"].dW
        grads["b1"] = self.layers["Affine1"].db
        grads["W2"] = self.layers["Affine2"].dW
        grads["b2"] = self.layers["Affine2"].db

        return grads
