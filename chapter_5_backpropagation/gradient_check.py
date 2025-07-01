import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from dataset import load_mnist
from two_layer_net import TwoLayerNet


(x_train, t_train), (x_test, t_test) = load_mnist(
    normalize=True, one_hot_label=True
)

network = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

# grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

# 각 가중치의 차이의 절댓값 계산 -> 그 평균을 계산 (수치미분이 너무오래걸려서 12시간 넘도록 계산이 안된다)
# for key in grad_numerical.keys():
#     diff = np.average(
#         np.abs(grad_backprop[key] - grad_numerical[key])
#     )
#     print(f"{key}: {diff}")

print(grad_backprop)
