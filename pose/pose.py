"""
人体姿态数据类 Pose。
包含关键点存储、角度计算、骨骼绘制及相似度匹配功能。
"""
import logging
import math
import os

import cv2
import numpy as np

from pose.keypoints import BODY_PARTS_KPT_IDS, BODY_PARTS_PAF_IDS
from pose.one_euro_filter import OneEuroFilter

logger = logging.getLogger(__name__)


class Pose:
    """
    单人体姿态对象，封装 18 个关键点及相关操作。

    Attributes:
        num_kpts (int):      关键点数量
        kpt_names (list):    关键点名称列表
        color (list):        骨骼绘制颜色 [B, G, R]
    """

    num_kpts = 18
    kpt_names = [
        'nose', 'neck',
        'r_sho', 'r_elb', 'r_wri',
        'l_sho', 'l_elb', 'l_wri',
        'r_hip', 'r_knee', 'r_ank',
        'l_hip', 'l_knee', 'l_ank',
        'r_eye', 'l_eye',
        'r_ear', 'l_ear',
    ]
    sigmas = np.array(
        [.26, .79, .79, .72, .62, .79, .72, .62,
         1.07, .87, .89, 1.07, .87, .89, .25, .25, .35, .35],
        dtype=np.float32
    ) / 10.0
    vars = (sigmas * 2) ** 2
    last_id = -1
    color = [0, 224, 255]

    def __init__(self, keypoints: np.ndarray, confidence: float):
        """
        Args:
            keypoints:  (18, 2) 关键点坐标数组，未检测到的为 -1
            confidence: 姿态整体置信度
        """
        super().__init__()
        self.keypoints = keypoints
        self.confidence = confidence
        self.bbox = Pose.get_bbox(self.keypoints)
        self.pose_action: str = None    # 'fallPeople' | 'normalPeople'
        self.action_fall: float = None
        self.action_normal: float = None
        self.img_pose: np.ndarray = None
        self.id: int = None
        self.filters = [[OneEuroFilter(), OneEuroFilter()] for _ in range(Pose.num_kpts)]

    @staticmethod
    def get_bbox(keypoints: np.ndarray) -> tuple:
        """
        根据有效关键点计算包围框。

        Args:
            keypoints: (18, 2) 关键点数组

        Returns:
            (x, y, w, h) 包围框
        """
        found = keypoints[keypoints[:, 0] != -1]
        if len(found) == 0:
            return (0, 0, 0, 0)
        return cv2.boundingRect(found.reshape(-1, 1, 2))

    def get_keypoints(self) -> np.ndarray:
        """
        提取有效关节点（用于行为分类网络输入）。

        Returns:
            (N, 2) 有效关键点数组，最多 13 个
        """
        assert self.keypoints.shape == (Pose.num_kpts, 2)
        points = []
        for part_id in range(len(BODY_PARTS_PAF_IDS) - 2):
            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
            if self.keypoints[kpt_b_id, 0] != -1:
                points.append(self.keypoints[kpt_b_id])
        return np.array(points)[:13]

    def cal_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """
        计算以 p2 为顶点，p1-p2-p3 构成的夹角（度数）。

        Args:
            p1, p2, p3: 关键点坐标 [x, y]，含 -1 则返回 0

        Returns:
            角度（0 ~ 180 度）
        """
        if any(v == -1 for pt in (p1, p2, p3) for v in pt):
            return 0
        a = math.sqrt((p2[0] - p3[0]) ** 2 + (p2[1] - p3[1]) ** 2)
        b = math.sqrt((p1[0] - p3[0]) ** 2 + (p1[1] - p3[1]) ** 2)
        c = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
        denom = -2 * a * c
        if denom == 0:
            return 0
        try:
            return math.degrees(math.acos((b * b - a * a - c * c) / denom))
        except ValueError:
            return 0

    def draw(self, img: np.ndarray, show_draw: bool = True):
        """
        在图像上绘制骨骼关节点和连线，并生成归一化骨骼图。

        Args:
            img:       输入图像（BGR）
            show_draw: 是否在原图上绘制关节点和骨骼线

        Returns:
            tuple: (骨骼图 128×128, 颈部 X, 颈部 Y, 左膝弯曲标志, 右膝弯曲标志)
        """
        assert self.keypoints.shape == (Pose.num_kpts, 2)

        left = right = 0

        # 膝盖角度检测（80~120° 判定为弯曲）
        left_angle  = self.cal_angle(self.keypoints[11], self.keypoints[12], self.keypoints[13])
        right_angle = self.cal_angle(self.keypoints[8],  self.keypoints[9],  self.keypoints[10])
        if 80 < left_angle < 120:
            left = 1
        if 80 < right_angle < 120:
            right = 1

        # 归一化骨骼图（128×128 灰度）
        iw, ih = self.bbox[2], self.bbox[3]
        I = np.zeros((128, 128), dtype=np.uint8)
        scale = 1.0
        if iw > 0 and ih > 0:
            scale = min(128.0 / iw, 128.0 / ih)

        for part_id in range(len(BODY_PARTS_PAF_IDS) - 2):
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
            kpt_a = self.keypoints[kpt_a_id]
            kpt_b = self.keypoints[kpt_b_id]

            has_a = kpt_a[0] != -1
            has_b = kpt_b[0] != -1

            if has_a:
                x_a, y_a = int(kpt_a[0]), int(kpt_a[1])
                if show_draw:
                    cv2.circle(img, (x_a, y_a), 3, Pose.color, -1)
                px_a = int((x_a - self.bbox[0]) * scale)
                py_a = int((y_a - self.bbox[1]) * scale)
                cv2.circle(I, (px_a, py_a), 3, 255, -1)

            if has_b:
                x_b, y_b = int(kpt_b[0]), int(kpt_b[1])
                if show_draw:
                    cv2.circle(img, (x_b, y_b), 3, Pose.color, -1)
                px_b = int((x_b - self.bbox[0]) * scale)
                py_b = int((y_b - self.bbox[1]) * scale)
                cv2.circle(I, (px_b, py_b), 3, 255, -1)

            if has_a and has_b:
                if show_draw:
                    cv2.line(img, (x_a, y_a), (x_b, y_b), Pose.color, 2)
                cv2.line(I, (px_a, py_a), (px_b, py_b), 255, 2)

        neck_x = int(self.keypoints[1][0])
        neck_y = int(self.keypoints[1][1])
        return I, neck_x, neck_y, left, right


# ─────────────────────────────────────────────
# 多帧姿态跟踪（相似度匹配）
# ─────────────────────────────────────────────

def get_similarity(a: "Pose", b: "Pose", threshold: float = 0.5) -> int:
    """
    计算两个姿态之间相似关键点数量（用于跨帧 ID 传播）。

    Args:
        a, b:      待比较的两个 Pose 对象
        threshold: 相似度阈值（基于指数距离）

    Returns:
        相似关键点数量
    """
    num_similar = 0
    for kpt_id in range(Pose.num_kpts):
        if a.keypoints[kpt_id, 0] != -1 and b.keypoints[kpt_id, 0] != -1:
            distance = np.sum((a.keypoints[kpt_id] - b.keypoints[kpt_id]) ** 2)
            area = max(a.bbox[2] * a.bbox[3], b.bbox[2] * b.bbox[3])
            similarity = np.exp(-distance / (2 * (area + np.spacing(1)) * Pose.vars[kpt_id]))
            if similarity > threshold:
                num_similar += 1
    return num_similar


def track_poses(previous_poses: list, current_poses: list,
                threshold: int = 3) -> None:
    """
    将上一帧的姿态 ID 传播至当前帧（贪心匹配）。

    Args:
        previous_poses: 上一帧姿态列表（含 id）
        current_poses:  当前帧姿态列表（待分配 id）
        threshold:      最小相似关键点数
    """
    current_poses = sorted(current_poses, key=lambda p: p.confidence, reverse=True)
    mask = np.ones(len(previous_poses), dtype=np.int32)

    for cur_pose in current_poses:
        best_id = None
        best_iou = 0
        best_idx = None

        for idx, prev_pose in enumerate(previous_poses):
            if not mask[idx]:
                continue
            iou = get_similarity(cur_pose, prev_pose)
            if iou > best_iou:
                best_iou = iou
                best_id = prev_pose.id
                best_idx = idx

        if best_iou >= threshold:
            mask[best_idx] = 0
        else:
            best_id = None

        if best_id is not None:
            cur_pose.id = best_id
        else:
            cur_pose.id = Pose.last_id + 1
            Pose.last_id += 1
