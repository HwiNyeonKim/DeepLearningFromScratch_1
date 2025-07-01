import numpy as np


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
