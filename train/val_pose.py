"""
OpenPose 在 COCO 验证集上的评估脚本。
"""
import argparse
import json
import logging
import math

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from datasets.coco import CocoValDataset
from models.with_mobilenet import PoseEstimationWithMobileNet
from pose.keypoints import extract_keypoints, group_keypoints
from pose.load_state import load_state
from config import VAL_LABELS_PATH, VAL_IMAGES_FOLDER, OUTPUT_DIR
import os

logger = logging.getLogger(__name__)


def run_coco_eval(gt_file: str, dt_file: str):
    """运行官方 COCO keypoints 评估。"""
    coco_gt = COCO(gt_file)
    coco_dt = coco_gt.loadRes(dt_file)
    result = COCOeval(coco_gt, coco_dt, 'keypoints')
    result.evaluate()
    result.accumulate()
    result.summarize()


def normalize(img, img_mean=(128, 128, 128), img_scale=1 / 256):
    return (np.array(img, dtype=np.float32) - img_mean) * img_scale


def pad_width(img, stride, pad_value, min_dims):
    h, w, _ = img.shape
    h = min(min_dims[0], h)
    min_dims[0] = math.ceil(min_dims[0] / stride) * stride
    min_dims[1] = max(min_dims[1], w)
    min_dims[1] = math.ceil(min_dims[1] / stride) * stride
    pad = [
        int(math.floor((min_dims[0] - h) / 2.0)),
        int(math.floor((min_dims[1] - w) / 2.0)),
        0, 0,
    ]
    pad[2] = int(min_dims[0] - h - pad[0])
    pad[3] = int(min_dims[1] - w - pad[1])
    padded = cv2.copyMakeBorder(img, pad[0], pad[2], pad[1], pad[3],
                                cv2.BORDER_CONSTANT, value=pad_value)
    return padded, pad


def infer(net, img, scales, base_height, stride, pad_value=(0, 0, 0),
          img_mean=(128, 128, 128), img_scale=1 / 256):
    normed = normalize(img, img_mean, img_scale)
    h, w, _ = normed.shape
    avg_heatmaps = np.zeros((h, w, 19), dtype=np.float32)
    avg_pafs     = np.zeros((h, w, 38), dtype=np.float32)

    for ratio in [scale * base_height / float(h) for scale in scales]:
        scaled = cv2.resize(normed, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
        min_dims = [base_height, max(scaled.shape[1], base_height)]
        padded, pad = pad_width(scaled, stride, pad_value, min_dims)

        tensor = torch.from_numpy(padded).permute(2, 0, 1).unsqueeze(0).float().cuda()
        with torch.no_grad():
            stages_output = net(tensor)

        hm = np.transpose(stages_output[-2].squeeze().cpu().numpy(), (1, 2, 0))
        hm = cv2.resize(hm, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        hm = hm[pad[0]:hm.shape[0] - pad[2], pad[1]:hm.shape[1] - pad[3]]
        hm = cv2.resize(hm, (w, h), interpolation=cv2.INTER_CUBIC)
        avg_heatmaps += hm / len(scales)

        pf = np.transpose(stages_output[-1].squeeze().cpu().numpy(), (1, 2, 0))
        pf = cv2.resize(pf, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        pf = pf[pad[0]:pf.shape[0] - pad[2], pad[1]:pf.shape[1] - pad[3]]
        pf = cv2.resize(pf, (w, h), interpolation=cv2.INTER_CUBIC)
        avg_pafs += pf / len(scales)

    return avg_heatmaps, avg_pafs


def evaluate(labels: str, output_name: str, images_folder: str, net,
             multiscale: bool = True, visualize: bool = False):
    """在 COCO 验证集上评估姿态估计模型。"""
    net = net.cuda().eval()
    scales = [0.5, 1.0, 1.5, 2.0] if multiscale else [1.0]
    stride = 8

    dataset = CocoValDataset(labels, images_folder)
    coco_result = []
    to_coco_map = [0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]

    for sample in dataset:
        img = sample['img']
        hm, pf = infer(net, img, scales, 368, stride)

        total_kpts = 0
        all_kpts_by_type = []
        for kpt_idx in range(18):
            total_kpts += extract_keypoints(hm[:, :, kpt_idx], all_kpts_by_type, total_kpts)

        pose_entries, all_keypoints = group_keypoints(all_kpts_by_type, pf)

        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            keypoints = [0] * 17 * 3
            position_id = -1
            person_score = pose_entries[n][-2]
            for keypoint_id in pose_entries[n][:-2]:
                position_id += 1
                if position_id == 1:
                    continue
                cx = cy = score = visibility = 0
                if keypoint_id != -1:
                    cx, cy, score = all_keypoints[int(keypoint_id), 0:3]
                    cx += 0.5
                    cy += 0.5
                    visibility = 1
                coco_idx = to_coco_map[position_id]
                if coco_idx >= 0:
                    keypoints[coco_idx * 3]     = cx
                    keypoints[coco_idx * 3 + 1] = cy
                    keypoints[coco_idx * 3 + 2] = visibility

            coco_result.append({
                'image_id':   int(sample['file_name'].rsplit('.', 1)[0]),
                'category_id': 1,
                'keypoints':  keypoints,
                'score': person_score * max(0, (pose_entries[n][-1] - 1)),
            })

            if visualize:
                for kp in [keypoints[i * 3: i * 3 + 3] for i in range(17)]:
                    cv2.circle(img, (int(kp[0]), int(kp[1])), 3, (255, 0, 255), -1)
                cv2.imshow('keypoints', img)
                if cv2.waitKey() == 27:
                    return

    os.makedirs(os.path.dirname(output_name), exist_ok=True)
    with open(output_name, 'w') as f:
        json.dump(coco_result, f, indent=4)

    run_coco_eval(labels, output_name)


def main():
    parser = argparse.ArgumentParser(description='OpenPose COCO 验证集评估')
    parser.add_argument('--labels',          default=VAL_LABELS_PATH)
    parser.add_argument('--output-name',     default=os.path.join(OUTPUT_DIR, 'detections.json'))
    parser.add_argument('--images-folder',   default=VAL_IMAGES_FOLDER)
    parser.add_argument('--checkpoint-path', default='weights/checkpoint_iter_370000.pth')
    parser.add_argument('--multiscale',      action='store_true', default=True)
    parser.add_argument('--visualize',       action='store_true', default=False)
    args = parser.parse_args()

    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load(args.checkpoint_path)
    load_state(net, checkpoint)
    evaluate(args.labels, args.output_name, args.images_folder,
             net, args.multiscale, args.visualize)


if __name__ == '__main__':
    main()
