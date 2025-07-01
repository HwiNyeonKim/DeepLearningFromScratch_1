import numpy as np


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
