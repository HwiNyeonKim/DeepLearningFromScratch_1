import os
import sys

import numpy as np
from PIL import Image

from dataset import load_mnist

# 경로를 먼저 추가
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


if __name__ == "__main__":
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

    img = x_train[0]
    label = t_train[0]
    print(label)  # 5

    print(img.shape)  # (784,)
    img = img.reshape(28, 28)
    print(img.shape)  # (28, 28)

    img_show(img)
