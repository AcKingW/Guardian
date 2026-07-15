"""
人体关键点提取与组合模块（OpenPose）
"""
import math
import numpy as np
from operator import itemgetter

# 身体部位 关键点对索引（PAF 关联的骨骼两端）
BODY_PARTS_KPT_IDS = [
    [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
    [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
    [1, 0], [0, 14], [14, 16], [0, 15], [15, 17], [2, 16], [5, 17],
]

# PAF（Part Affinity Fields）通道索引
BODY_PARTS_PAF_IDS = (
    [12, 13], [20, 21], [14, 15], [16, 17], [22, 23], [24, 25],
    [0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11],
    [28, 29], [30, 31], [34, 35], [32, 33], [36, 37], [18, 19], [26, 27],
)


def linspace2d(start, stop, n=10):
    """在 start 到 stop 之间生成 n 个等间隔的二维点。"""
    points = 1 / (n - 1) * (stop - start)
    return points[:, None] * np.arange(n) + start[:, None]


def extract_keypoints(heatmap, all_keypoints, total_keypoint_num):
    """
    从单个关节热图中提取局部极大值作为关键点候选。

    Args:
        heatmap: 单通道热图 (H, W)
        all_keypoints: 累积关键点列表（就地追加）
        total_keypoint_num: 已有关键点总数（用于分配全局 ID）

    Returns:
        本次提取的关键点数量
    """
    heatmap[heatmap < 0.1] = 0
    heatmap_with_borders = np.pad(heatmap, [(2, 2), (2, 2)], mode='constant')

    hc = heatmap_with_borders[1:-1, 1:-1]
    hl = heatmap_with_borders[1:-1, 2:]
    hr = heatmap_with_borders[1:-1, 0:-2]
    hu = heatmap_with_borders[2:, 1:-1]
    hd = heatmap_with_borders[0:-2, 1:-1]

    heatmap_peaks = (hc > hl) & (hc > hr) & (hc > hu) & (hc > hd)
    heatmap_peaks = heatmap_peaks[1:-1, 1:-1]

    keypoints = list(zip(np.nonzero(heatmap_peaks)[1], np.nonzero(heatmap_peaks)[0]))
    keypoints = sorted(keypoints, key=itemgetter(0))

    suppressed = np.zeros(len(keypoints), dtype=np.uint8)
    keypoints_with_score_and_id = []
    keypoint_num = 0

    for i in range(len(keypoints)):
        if suppressed[i]:
            continue
        for j in range(i + 1, len(keypoints)):
            if math.sqrt(
                (keypoints[i][0] - keypoints[j][0]) ** 2 +
                (keypoints[i][1] - keypoints[j][1]) ** 2
            ) < 6:
                suppressed[j] = 1
        kpt = (
            keypoints[i][0], keypoints[i][1],
            heatmap[keypoints[i][1], keypoints[i][0]],
            total_keypoint_num + keypoint_num,
        )
        keypoints_with_score_and_id.append(kpt)
        keypoint_num += 1

    all_keypoints.append(keypoints_with_score_and_id)
    return keypoint_num


def group_keypoints(all_keypoints_by_type, pafs, pose_entry_size=20,
                    min_paf_score=0.05, demo=False):
    """
    利用 PAF 将候选关键点组合成完整的人体姿态。

    Args:
        all_keypoints_by_type: 每类关键点的候选列表
        pafs: Part Affinity Fields (H, W, C)
        pose_entry_size: 姿态编码长度（18 关节 + 得分 + 计数）
        min_paf_score: PAF 连接最小评分
        demo: 是否为演示模式（影响阈值）

    Returns:
        pose_entries: 组合后的姿态条目列表
        all_keypoints: 全局关键点数组 (N, 4) [x, y, score, id]
    """
    all_keypoints = np.array([item for sublist in all_keypoints_by_type for item in sublist])
    pose_entries = []

    for part_id in range(len(BODY_PARTS_PAF_IDS)):
        part_pafs = pafs[:, :, BODY_PARTS_PAF_IDS[part_id]]
        kpts_a = all_keypoints_by_type[BODY_PARTS_KPT_IDS[part_id][0]]
        kpts_b = all_keypoints_by_type[BODY_PARTS_KPT_IDS[part_id][1]]
        n, m = len(kpts_a), len(kpts_b)
        if n == 0 or m == 0:
            continue

        connections = []
        score_mid = part_pafs
        for i, kpt_a in enumerate(kpts_a):
            for j, kpt_b in enumerate(kpts_b):
                mid_point = [
                    (int(round((kpt_a[0] + kpt_b[0]) * 0.5)),
                     int(round((kpt_a[1] + kpt_b[1]) * 0.5))),
                ]
                vec = [kpt_b[0] - kpt_a[0], kpt_b[1] - kpt_a[1]]
                vec_norm = math.sqrt(vec[0] ** 2 + vec[1] ** 2)
                if vec_norm == 0:
                    continue
                vec = [vec[0] / vec_norm, vec[1] / vec_norm]

                num_inter = 10
                x_points = np.round(linspace2d(
                    np.array([kpt_a[0], kpt_a[1]]),
                    np.array([kpt_b[0], kpt_b[1]]),
                    num=num_inter,
                )).astype(np.int32)

                paf_scores = np.zeros(num_inter)
                for k in range(num_inter):
                    x = min(max(x_points[0, k], 0), score_mid.shape[1] - 1)
                    y = min(max(x_points[1, k], 0), score_mid.shape[0] - 1)
                    paf_scores[k] = (
                        score_mid[y, x, 0] * vec[0] +
                        score_mid[y, x, 1] * vec[1]
                    )

                in_threshold = paf_scores > min_paf_score
                paf_score = np.sum(paf_scores[in_threshold])
                criterion1 = np.sum(in_threshold) > 0.8 * num_inter
                criterion2 = paf_score > 0
                if criterion1 and criterion2:
                    connections.append([i, j, paf_score, paf_score + kpt_a[2] + kpt_b[2]])

        if not connections:
            continue

        connections = sorted(connections, key=lambda x: x[2], reverse=True)
        used_idx_a, used_idx_b = set(), set()
        valid_connections = []
        for conn in connections:
            i, j = conn[0], conn[1]
            if i in used_idx_a or j in used_idx_b:
                continue
            valid_connections.append([
                kpts_a[i][3], kpts_b[j][3], conn[2]
            ])
            used_idx_a.add(i)
            used_idx_b.add(j)

        for conn in valid_connections:
            found = False
            for pe in pose_entries:
                if (pe[BODY_PARTS_KPT_IDS[part_id][0]] == conn[0] or
                        pe[BODY_PARTS_KPT_IDS[part_id][1]] == conn[1]):
                    pe[BODY_PARTS_KPT_IDS[part_id][0]] = conn[0]
                    pe[BODY_PARTS_KPT_IDS[part_id][1]] = conn[1]
                    pe[-1] += 1
                    pe[-2] += all_keypoints[conn[1]][2] + conn[2]
                    found = True
                    break
            if not found:
                pe = np.full(pose_entry_size, -1, dtype=np.float32)
                pe[BODY_PARTS_KPT_IDS[part_id][0]] = conn[0]
                pe[BODY_PARTS_KPT_IDS[part_id][1]] = conn[1]
                pe[-1] = 2
                pe[-2] = (
                    all_keypoints[conn[0]][2] +
                    all_keypoints[conn[1]][2] +
                    conn[2]
                )
                pose_entries.append(pe)

    return pose_entries, all_keypoints
