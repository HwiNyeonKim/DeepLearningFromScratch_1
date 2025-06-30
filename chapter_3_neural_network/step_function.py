import numpy as np
from matplotlib import pyplot as plt


def step_function(x):
    y = x > 0
    return y.astype(np.int64)


# print(step_function(np.array([-1.0, 1.0, 2.0])))
# print(type(step_function(np.array([-1.0, 1.0, 2.0]))))

if __name__ == "__main__":
    x = np.arange(-5.0, 5.0, 0.1)
    y = step_function(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()
