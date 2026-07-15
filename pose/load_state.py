"""
模型权重加载工具。
支持从检查点直接加载，以及从 MobileNet 预训练权重迁移。
"""
import collections
import logging
import torch
from torch import nn

logger = logging.getLogger(__name__)


def load_state(net: nn.Module, checkpoint: dict) -> None:
    """
    从检查点安全地加载模型权重。
    形状不匹配的层保留原始权重并打印警告。

    Args:
        net:        目标模型
        checkpoint: 包含 'state_dict' 键的检查点字典
    """
    source_state = checkpoint['state_dict']
    target_state = net.state_dict()
    new_target_state = collections.OrderedDict()

    for target_key, target_value in target_state.items():
        if (target_key in source_state and
                source_state[target_key].size() == target_value.size()):
            new_target_state[target_key] = source_state[target_key]
        else:
            new_target_state[target_key] = target_value
            logger.warning("未找到预训练参数: %s", target_key)

    net.load_state_dict(new_target_state)


def load_from_mobilenet(net: nn.Module, checkpoint: dict) -> None:
    """
    从 MobileNet 预训练检查点迁移权重至目标模型。
    自动处理 'model' → 'module.model' 的键名映射。

    Args:
        net:        目标模型
        checkpoint: MobileNet 检查点字典
    """
    source_state = checkpoint['state_dict']
    target_state = net.state_dict()
    new_target_state = collections.OrderedDict()

    for target_key, target_value in target_state.items():
        k = target_key.replace('model', 'module.model') if 'model' in target_key else target_key
        if k in source_state and source_state[k].size() == target_value.size():
            new_target_state[target_key] = source_state[k]
        else:
            new_target_state[target_key] = target_value
            logger.warning("未找到预训练参数: %s", target_key)

    net.load_state_dict(new_target_state)
