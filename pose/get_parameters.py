"""
从 PyTorch 模型中提取指定类型的参数（用于构建分组优化器）。
"""
from typing import Generator
from torch import nn


def get_parameters(model: nn.Module, predicate) -> Generator:
    """
    遍历模型中所有模块，按条件 predicate(module, param_name) 筛选参数。

    Args:
        model:     PyTorch 模型
        predicate: 接受 (module, param_name) 返回 bool 的函数

    Yields:
        满足条件的参数张量
    """
    for module in model.modules():
        for param_name, param in module.named_parameters():
            if predicate(module, param_name):
                yield param


def get_parameters_conv(model: nn.Module, name: str) -> Generator:
    """提取标准 Conv2d（groups=1）的指定参数（weight 或 bias）。"""
    return get_parameters(
        model,
        lambda m, p: isinstance(m, nn.Conv2d) and m.groups == 1 and p == name,
    )


def get_parameters_conv_depthwise(model: nn.Module, name: str) -> Generator:
    """提取深度可分离 Conv2d（groups == in_channels == out_channels）的参数。"""
    return get_parameters(
        model,
        lambda m, p: (
            isinstance(m, nn.Conv2d)
            and m.groups == m.in_channels
            and m.in_channels == m.out_channels
            and p == name
        ),
    )


def get_parameters_bn(model: nn.Module, name: str) -> Generator:
    """提取 BatchNorm2d 的指定参数（weight 或 bias）。"""
    return get_parameters(
        model,
        lambda m, p: isinstance(m, nn.BatchNorm2d) and p == name,
    )
