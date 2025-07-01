from chapter_3_neural_network import cross_entropy_error, softmax

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None  # loss function
        self.y = None  # output of softmax
        self.t = None  # one-hot vector of ground truth

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        if self.t is None:
            raise ValueError("t is not set yet.")

        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size  # 데이터 1개당 오차를 역전파

        return dx
