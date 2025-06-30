import numpy as np


def numerical_gradient(f, x):
    """
    수치 미분을 통해 기울기를 계산하는 함수

    Args:
        f (function): 미분하고자 하는 함수
        x (numpy.ndarray): 미분하고자 하는 함수의 입력값

    Returns:
        numpy.ndarray: 함수 f의 입력값 x에 대한 기울기
    """
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            tmp_val = x[idx]
            x[idx] = tmp_val + h
            fxh1 = f(x)

            x[idx] = tmp_val - h
            fxh2 = f(x)

            grad[idx] = (fxh1 - fxh2) / (2 * h)
            x[idx] = tmp_val
            it.iternext()

    return grad
