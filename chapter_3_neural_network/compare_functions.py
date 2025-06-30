import matplotlib.pylab as plt
import numpy as np
from sigmoid_function import sigmoid
from step_function import step_function

if __name__ == "__main__":
    x = np.arange(-5.0, 5.0, 0.1)
    y1 = step_function(x)
    y2 = sigmoid(x)

    plt.plot(x, y1, label="step_function", linestyle="--")
    plt.plot(x, y2, label="sigmoid")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("step_function and sigmoid")
    plt.show()
