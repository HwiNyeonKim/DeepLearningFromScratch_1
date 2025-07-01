class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        """
        Args:
            x (numpy.ndarray): 입력 데이터

        Returns:
            numpy.ndarray: 출력 데이터
        """
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        """
        Args:
            dout (numpy.ndarray): 출력 데이터의 미분값

        Returns:
            numpy.ndarray: 입력 데이터의 미분값 (0 또는 입력값)
            - 입력값이 0보다 같거나 작으면 0, 0보다 크면 입력값 그대로 반환
        """
        dout[self.mask] = 0
        dx = dout

        return dx
