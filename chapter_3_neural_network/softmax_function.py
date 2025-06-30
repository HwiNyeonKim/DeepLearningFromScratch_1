import numpy as np


def softmax(a):
    if a.ndim == 2:
        c = np.max(a, axis=1, keepdims=True)
        exp_a = np.exp(a - c)
        sum_exp_a = np.sum(exp_a, axis=1, keepdims=True)
        y = exp_a / sum_exp_a
        return y

    c = np.max(a)  # 오버플로우 방지를 위해 최대값을 빼줌
    a = a - c
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


if __name__ == "__main__":
    a = np.array([0.3, 2.9, 4.0])
    y = softmax(a)
    print(f"original values: {y}")
    print(f"sum of values: {np.sum(y)}")

    y2 = softmax(a + 1000)
    print(f"values after adding 1000: {y2}")
    print(f"sum of values: {np.sum(y2)}")
