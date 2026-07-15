"""
姿态估计损失函数。
"""
import torch
from torch import Tensor


def l2_loss(prediction: Tensor, target: Tensor,
            mask: Tensor, batch_size: int) -> Tensor:
    """
    带掩码的 L2 损失（用于 OpenPose 热图和 PAF 监督）。

    Args:
        prediction: 模型预测输出
        target:     真实标签
        mask:       有效区域掩码（0 处不计算损失）
        batch_size: 批大小（用于归一化）

    Returns:
        标量损失值
    """
    loss = (prediction - target) * mask
    loss = (loss * loss) / 2 / batch_size
    return loss.sum()
