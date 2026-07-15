"""
OpenPose 推理管线。
负责从帧提供者读取图像 → 姿态估计 → 行为分类 → 可视化展示。
"""
import logging
from math import ceil, floor
from timeit import default_timer as timer

import cv2
import numpy as np
import torch
from torch import jit

from config import (
    POSE_CHECKPOINT_PATH, ACTION_CHECKPOINT_PATH,
    POSE_HEIGHT_SIZE, POSE_STRIDE, POSE_UPSAMPLE_RATIO, POSE_CPU_MODE,
    JUMP_HEIGHT_DIFF_THRESHOLD, JUMP_DISPLAY_DURATION,
)
from pose.keypoints import extract_keypoints, group_keypoints
from pose.pose import Pose
from action.detect import action_detect
from pipeline.utils import cv2_add_chinese_text
from pipeline.video_reader import build_frame_provider

logger = logging.getLogger(__name__)

# 全局状态（跳跃/跌倒统计）
_jump_ok: int = 0
_fall_ok: int = 0


def normalize(img: np.ndarray, img_mean=(128, 128, 128), img_scale=1 / 256) -> np.ndarray:
    """图像归一化至 [-0.5, 0.5) 区间。"""
    return (np.array(img, dtype=np.float32) - img_mean) * img_scale


def pad_width(img: np.ndarray, stride: int, pad_value: tuple, min_dims: list):
    """对图像进行对称填充使尺寸能被 stride 整除。"""
    h, w, _ = img.shape
    h = min(min_dims[0], h)
    min_dims[0] = ceil(min_dims[0] / float(stride)) * stride
    min_dims[1] = max(min_dims[1], w)
    min_dims[1] = ceil(min_dims[1] / float(stride)) * stride
    pad = [
        int(floor((min_dims[0] - h) / 2.0)),
        int(floor((min_dims[1] - w) / 2.0)),
        0, 0,
    ]
    pad[2] = int(min_dims[0] - h - pad[0])
    pad[3] = int(min_dims[1] - w - pad[1])
    padded = cv2.copyMakeBorder(
        img, pad[0], pad[2], pad[1], pad[3],
        cv2.BORDER_CONSTANT, value=pad_value,
    )
    return padded, pad


def infer_fast(net, img: np.ndarray, height_size: int, stride: int,
               upsample_ratio: int, cpu: bool,
               pad_value=(0, 0, 0), img_mean=(128, 128, 128), img_scale=1 / 256):
    """
    快速姿态估计推理（单尺度）。

    Returns:
        heatmaps: 关节热图 (H, W, 19)
        pafs:     PAF (H, W, 38)
        scale:    缩放比例
        pad:      填充量 [top, left, bottom, right]
    """
    h, w, _ = img.shape
    scale = height_size / h
    scaled = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    scaled = normalize(scaled, img_mean, img_scale)
    min_dims = [height_size, max(scaled.shape[1], height_size)]
    padded, pad = pad_width(scaled, stride, pad_value, min_dims)

    tensor = torch.from_numpy(padded).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor = tensor.cuda()

    with torch.no_grad():
        stages_output = net(tensor)

    heatmaps = np.transpose(
        stages_output[-2].squeeze().cpu().numpy(), (1, 2, 0)
    )
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio,
                          interpolation=cv2.INTER_CUBIC)

    pafs = np.transpose(
        stages_output[-1].squeeze().cpu().numpy(), (1, 2, 0)
    )
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio,
                      interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad


def run_detection(net, action_net, frame_provider,
                  height_size: int = POSE_HEIGHT_SIZE,
                  cpu: bool = POSE_CPU_MODE,
                  window_title: str = 'Guardian - Pose Detection'):
    """
    主检测循环：读帧 → 姿态估计 → 行为分类 → 展示。

    Args:
        net:            姿态估计网络（JIT 模型）
        action_net:     行为分类网络（JIT 模型）
        frame_provider: 可迭代帧提供者
        height_size:    网络输入高度
        cpu:            是否强制 CPU 模式
        window_title:   显示窗口标题
    """
    global _jump_ok, _fall_ok

    net = net.eval()
    if not cpu:
        net = net.cuda()

    num_keypoints = Pose.num_kpts
    last_height = 0
    duration_jump = 0
    duration_fall = 0

    for img in frame_provider:
        orig_img = img.copy()

        heatmaps, pafs, scale, pad = infer_fast(
            net, img, height_size, POSE_STRIDE, POSE_UPSAMPLE_RATIO, cpu
        )

        # ── 关键点提取 ─────────────────────────────
        total_kpts = 0
        all_kpts_by_type = []
        for kpt_idx in range(num_keypoints):
            total_kpts += extract_keypoints(heatmaps[:, :, kpt_idx], all_kpts_by_type, total_kpts)

        pose_entries, all_keypoints = group_keypoints(all_kpts_by_type, pafs, demo=True)

        # ── 坐标还原到原始尺寸 ─────────────────────
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (
                all_keypoints[kpt_id, 0] * POSE_STRIDE / POSE_UPSAMPLE_RATIO - pad[1]
            ) / scale
            all_keypoints[kpt_id, 1] = (
                all_keypoints[kpt_id, 1] * POSE_STRIDE / POSE_UPSAMPLE_RATIO - pad[0]
            ) / scale

        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_kpts = np.full((num_keypoints, 2), -1, dtype=np.int32)
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:
                    pose_kpts[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_kpts[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = Pose(pose_kpts, pose_entries[n][18])
            if len(pose.get_keypoints()) >= 10:
                current_poses.append(pose)

        # ── 字幕展示 ──────────────────────────────
        if duration_jump > 0:
            img = cv2_add_chinese_text(img, '危险行为:跳跃', 10, 40, (255, 0, 0), 50)
            duration_jump -= 1
        if duration_fall > 0:
            img = cv2_add_chinese_text(img, '异常行为:跌倒', 10, 100, (255, 0, 0), 50)
            duration_fall -= 1

        # ── 逐人分析 ──────────────────────────────
        for pose in current_poses:
            pose.img_pose, cur_x, cur_height, left, right = pose.draw(img)

            # 跳跃检测
            if cur_height < last_height and (last_height - cur_height) > JUMP_HEIGHT_DIFF_THRESHOLD:
                img = cv2_add_chinese_text(img, '危险行为:跳跃', 10, 40, (255, 0, 0), 50)
                _jump_ok = 1
                duration_jump = JUMP_DISPLAY_DURATION
            last_height = cur_height

            # 行为分类（跌倒检测）
            crown_proportion = pose.bbox[2] / max(pose.bbox[3], 1)
            pose = action_detect(action_net, pose, crown_proportion)

            color = (0, 0, 255) if pose.pose_action == 'fallPeople' else (0, 255, 0)
            if pose.pose_action == 'fallPeople':
                _fall_ok = 1
                duration_fall = JUMP_DISPLAY_DURATION

            cv2.rectangle(
                img,
                (pose.bbox[0], pose.bbox[1]),
                (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]),
                color, 2 if pose.pose_action == 'fallPeople' else 1,
            )
            cv2.putText(
                img, f'state: {pose.pose_action}',
                (pose.bbox[0], pose.bbox[1] - 16),
                cv2.FONT_HERSHEY_COMPLEX, 0.5, color,
            )

        img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
        cv2.imshow(window_title, img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def load_and_run(video_source: str = '', image_source: str = '',
                 video_name: str = 'Guardian',
                 pose_checkpoint: str = POSE_CHECKPOINT_PATH,
                 action_checkpoint: str = ACTION_CHECKPOINT_PATH,
                 cpu: bool = POSE_CPU_MODE):
    """
    加载模型并启动检测。

    Args:
        video_source:      视频路径 / 摄像头 ID
        image_source:      图片路径 / 目录
        video_name:        监控点名称（叠加到画面）
        pose_checkpoint:   OpenPose JIT 权重路径
        action_checkpoint: 行为分类 JIT 权重路径
        cpu:               是否强制 CPU 推理
    """
    net = jit.load(pose_checkpoint)
    action_net = jit.load(action_checkpoint)
    frame_provider = build_frame_provider(video_source, image_source, video_name)
    run_detection(net, action_net, frame_provider, cpu=cpu)
