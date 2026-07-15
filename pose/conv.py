"""
可组合的 PyTorch 卷积模块工厂函数。
"""
from torch import nn


def conv(in_channels: int, out_channels: int, kernel_size: int = 3,
         padding: int = 1, bn: bool = True, dilation: int = 1,
         stride: int = 1, relu: bool = True, bias: bool = True) -> nn.Sequential:
    """
    构建标准卷积块：Conv2d → [BatchNorm2d] → [ReLU]。

    Args:
        in_channels:  输入通道数
        out_channels: 输出通道数
        kernel_size:  卷积核大小
        padding:      填充大小
        bn:           是否使用 BatchNorm
        dilation:     空洞卷积膨胀率
        stride:       步长
        relu:         是否使用 ReLU 激活
        bias:         是否使用偏置

    Returns:
        nn.Sequential 卷积块
    """
    modules = [nn.Conv2d(in_channels, out_channels, kernel_size,
                         stride, padding, dilation, bias=bias)]
    if bn:
        modules.append(nn.BatchNorm2d(out_channels))
    if relu:
        modules.append(nn.ReLU(inplace=True))
    return nn.Sequential(*modules)


def conv_dw(in_channels: int, out_channels: int, kernel_size: int = 3,
            padding: int = 1, stride: int = 1, dilation: int = 1) -> nn.Sequential:
    """深度可分离卷积（Depthwise + Pointwise），含 BN 和 ReLU。"""
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding,
                  dilation=dilation, groups=in_channels, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def conv_dw_no_bn(in_channels: int, out_channels: int, kernel_size: int = 3,
                  padding: int = 1, stride: int = 1, dilation: int = 1) -> nn.Sequential:
    """深度可分离卷积（无 BN），使用 ELU 激活。"""
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding,
                  dilation=dilation, groups=in_channels, bias=False),
        nn.ELU(inplace=True),
        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.ELU(inplace=True),
    )
