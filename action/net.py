"""
行为分类神经网络定义（跌倒 / 正常状态）。
提供多种网络架构供训练和推理使用。
"""
from torch import nn, randn, exp, sum as tsum


class NetV1(nn.Module):
    """最简线性分类器（权重矩阵直接 softmax）。"""

    def __init__(self):
        super().__init__()
        self.W = nn.Parameter(randn(16384, 2))

    def forward(self, x):
        h = x @ self.W
        h = exp(h)
        return h / tsum(h, dim=1, keepdim=True)


class NetV2(nn.Module):
    """两层全连接网络 + Softmax（推荐用于推理）。"""

    def __init__(self):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Linear(16384, 100),
            nn.ReLU(),
            nn.Linear(100, 2),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.sequential(x)


class SimpleNet(nn.Module):
    """三层全连接网络（无激活函数）。"""

    def __init__(self, in_dim: int, n_hidden_1: int, n_hidden_2: int, out_dim: int):
        super().__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, out_dim)

    def forward(self, x):
        return self.layer3(self.layer2(self.layer1(x)))


class ActivationNet(nn.Module):
    """三层全连接网络 + ReLU 激活。"""

    def __init__(self, in_dim: int, n_hidden_1: int, n_hidden_2: int, out_dim: int):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Linear(n_hidden_2, out_dim)

    def forward(self, x):
        return self.layer3(self.layer2(self.layer1(x)))


class BatchNet(nn.Module):
    """三层全连接网络 + BatchNorm + ReLU（收敛最快）。"""

    def __init__(self, in_dim: int, n_hidden_1: int, n_hidden_2: int, out_dim: int):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1),
            nn.BatchNorm1d(n_hidden_1),
            nn.ReLU(True),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.BatchNorm1d(n_hidden_2),
            nn.ReLU(True),
        )
        self.layer3 = nn.Linear(n_hidden_2, out_dim)

    def forward(self, x):
        return self.layer3(self.layer2(self.layer1(x)))
