import numpy as np


def step_function(x):
    return np.array(x > 0, dtype=np.int64)


def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))


def relu_function(x):
    return np.maximum(0, x)


def softmax_function(a):
    # 입력이 2차원일 때, 즉 배치 처리를 하는 경우
    if a.ndim == 2:
        a = a.T
        a = a - np.max(a, axis=0)
        y = np.exp(a) / np.sum(np.exp(a), axis=0)
        return y.T

    # 입력이 1차원일 때, 즉 단일 샘플에 대해 softmax를 적용하는 경우
    c = np.max(a)  # 오버플로우 방지를 위해 최대값을 빼줌
    a = a - c
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def cross_entropy_error_function(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size
