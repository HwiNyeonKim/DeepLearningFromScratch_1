import matplotlib.pylab as plt
import numpy as np


def relu(x):
    """ReLU function"""
    return np.maximum(0, x)


if __name__ == "__main__":
    x = np.arange(-5.0, 5.0, 0.1)
    y = relu(x)
    plt.plot(x, y)
    plt.show()
