import numpy as np
from .functions import softmax_function, cross_entropy_error_function


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        """
        Args:
            x (numpy.ndarray): 입력 데이터

        Returns:
            numpy.ndarray: 출력 데이터
        """
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout):
        """
        Args:
            dout (numpy.ndarray): 출력 데이터의 미분값

        Returns:
            numpy.ndarray: 입력 데이터의 미분값
        """
        if self.x is None:
            raise ValueError("x is not set yet.")

        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        """
        Args:
            x (numpy.ndarray): 입력 데이터

        Returns:
            numpy.ndarray: 출력 데이터
        """
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    def backward(self, dout):
        """
        Args:
            dout (numpy.ndarray): 출력 데이터의 미분값

        Returns:
            numpy.ndarray: 입력 데이터의 미분값
        """
        if self.out is None:
            raise ValueError("out is not set yet.")

        dx = dout * (1.0 - self.out) * self.out

        return dx


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        """
        Args:
            x (numpy.ndarray): 입력 데이터

        Returns:
            numpy.ndarray: 출력 데이터
        """
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        """
        Args:
            dout (numpy.ndarray): 출력 데이터의 미분값

        Returns:
            numpy.ndarray: 입력 데이터의 미분값 (0 또는 입력값)
            - 입력값이 0보다 같거나 작으면 0, 0보다 크면 입력값 그대로 반환
        """
        dout[self.mask] = 0
        dx = dout

        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax_function(x)
        self.loss = cross_entropy_error_function(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        if self.t is None:
            raise ValueError("t is not set yet.")

        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx
