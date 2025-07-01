class MultiplyLayer:
    def __init__(self):
        # 순전파시의 입력값 유지를 위한 필드
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        output = x * y

        return output

    def backward(self, dout):
        """
        Args:
            dout (float): 출력한 데이터의 미분값

        Returns:
            tuple: 출력했던 데이터의 미분값에 순전파때의 값을 서로 바꿔 곱한 값
        """
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy
