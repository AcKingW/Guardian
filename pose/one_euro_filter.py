"""
One Euro Filter — 用于平滑关键点抖动的低通滤波器。
原始论文: "Reducing latency for hand tracking in interactive systems"
"""
from math import pi


def get_alpha(rate: float = 30, cutoff: float = 1) -> float:
    """根据采样率和截止频率计算平滑系数 alpha。"""
    tau = 1 / (2 * pi * cutoff)
    te = 1 / rate
    return 1 / (1 + tau / te)


class LowPassFilter:
    """一阶低通滤波器。"""

    def __init__(self):
        self.x_previous = None

    def __call__(self, x: float, alpha: float = 0.5) -> float:
        if self.x_previous is None:
            self.x_previous = x
            return x
        x_filtered = alpha * x + (1 - alpha) * self.x_previous
        self.x_previous = x_filtered
        return x_filtered


class OneEuroFilter:
    """
    One Euro Filter：自适应低通滤波，高速运动时减少延迟，
    静止时有效抑制抖动。

    Args:
        freq:      采样频率 (Hz)
        mincutoff: 最小截止频率
        beta:      速度系数（越大对高速运动响应越快）
        dcutoff:   导数截止频率
    """

    def __init__(self, freq: float = 15, mincutoff: float = 1,
                 beta: float = 0.05, dcutoff: float = 1):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.filter_x = LowPassFilter()
        self.filter_dx = LowPassFilter()
        self.x_previous = None
        self.dx = None

    def __call__(self, x: float) -> float:
        if self.dx is None:
            self.dx = 0
        else:
            self.dx = (x - self.x_previous) * self.freq
        dx_smoothed = self.filter_dx(self.dx, get_alpha(self.freq, self.dcutoff))
        cutoff = self.mincutoff + self.beta * abs(dx_smoothed)
        x_filtered = self.filter_x(x, get_alpha(self.freq, cutoff))
        self.x_previous = x
        return x_filtered
