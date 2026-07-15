"""
行为分类推理模块。
接收骨骼图和宽高比，输出行为标签（跌倒/正常）。
"""
import numpy as np
import torch
from torch import from_numpy, argmax

from config import FALL_PROBABILITY_THRESHOLD


def action_detect(net: torch.nn.Module, pose, crown_proportion: float):
    """
    对单帧单人骨骼图执行行为分类推理。

    将骨骼图（128×128 灰度）展平后送入分类网络，
    融合宽高比特征（crown_proportion）加权计算最终跌倒概率。

    Args:
        net:              行为分类网络（已加载权重）
        pose:             Pose 对象（含 img_pose 属性）
        crown_proportion: 人体包围框宽高比（宽/高），躺倒时 > 1

    Returns:
        pose:  更新了 pose_action / action_fall / action_normal 的 Pose 对象
    """
    img = pose.img_pose.reshape(-1).astype(np.float32) / 255.0
    tensor = from_numpy(img[None, :]).cpu()

    with torch.no_grad():
        prediction = net(tensor)

    action_id = int(argmax(prediction, dim=1).cpu().item())

    # 融合宽高比：横卧时 (crown_proportion-1) 贡献正值提升跌倒概率
    possible_rate = (
        0.6 * prediction[:, action_id] +
        0.4 * (crown_proportion - 1)
    ).detach().numpy()[0]

    possible_rate = float(np.clip(possible_rate, 0.0, 1.0))

    if possible_rate > FALL_PROBABILITY_THRESHOLD:
        pose.pose_action = 'fallPeople'
        pose.action_fall = possible_rate
        pose.action_normal = 1.0 - possible_rate
    else:
        pose.pose_action = 'normalPeople'
        pose.action_fall = possible_rate
        pose.action_normal = 1.0 - possible_rate

    return pose
