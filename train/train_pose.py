"""
OpenPose 轻量化模型训练脚本。
使用方式:
    python -m train.train_pose --train-images-folder /path/to/coco/train2017
"""
import argparse
import logging
import os

import cv2
import torch
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.coco import CocoTrainDataset
from datasets.transformations import ConvertKeypoints, Scale, Rotate, CropPad, Flip
from models.with_mobilenet import PoseEstimationWithMobileNet
from pose.get_parameters import (
    get_parameters_conv, get_parameters_bn, get_parameters_conv_depthwise
)
from pose.loss import l2_loss
from pose.load_state import load_state, load_from_mobilenet
from train.val_pose import evaluate
from config import (
    TRAIN_LABELS_PATH, TRAIN_IMAGES_FOLDER, VAL_LABELS_PATH, VAL_IMAGES_FOLDER,
    TRAIN_BASE_LR, TRAIN_BATCH_SIZE, TRAIN_NUM_WORKERS, TRAIN_REFINEMENT_STAGES,
    CHECKPOINTS_DIR, OUTPUT_DIR,
)

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)  # 防止 DataLoader 死锁

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def train(prepared_train_labels: str, train_images_folder: str,
          num_refinement_stages: int = TRAIN_REFINEMENT_STAGES,
          base_lr: float = TRAIN_BASE_LR,
          batch_size: int = TRAIN_BATCH_SIZE,
          batches_per_iter: int = 1,
          num_workers: int = TRAIN_NUM_WORKERS,
          checkpoint_path: str = None,
          weights_only: bool = False,
          from_mobilenet: bool = False,
          checkpoints_folder: str = CHECKPOINTS_DIR,
          log_after: int = 100,
          val_labels: str = VAL_LABELS_PATH,
          val_images_folder: str = VAL_IMAGES_FOLDER,
          val_output_name: str = None,
          checkpoint_after: int = 5000,
          val_after: int = 5000):

    if val_output_name is None:
        val_output_name = os.path.join(OUTPUT_DIR, 'detections.json')

    net = PoseEstimationWithMobileNet(num_refinement_stages).cuda()

    dataset = CocoTrainDataset(
        prepared_train_labels, train_images_folder,
        stride=8, sigma=7, path_thickness=1,
        transform=transforms.Compose([
            ConvertKeypoints(), Scale(), Rotate(pad=(128, 128, 128)),
            CropPad(pad=(128, 128, 128)), Flip(),
        ]),
    )
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    optimizer = optim.Adam([
        {'params': get_parameters_conv(net.model, 'weight')},
        {'params': get_parameters_conv_depthwise(net.model, 'weight'), 'weight_decay': 0},
        {'params': get_parameters_bn(net.model, 'weight'), 'weight_decay': 0},
        {'params': get_parameters_bn(net.model, 'bias'), 'lr': base_lr * 2, 'weight_decay': 0},
        {'params': get_parameters_conv(net.cpm, 'weight'), 'lr': base_lr},
        {'params': get_parameters_conv(net.cpm, 'bias'), 'lr': base_lr * 2, 'weight_decay': 0},
        {'params': get_parameters_conv_depthwise(net.cpm, 'weight'), 'weight_decay': 0},
        {'params': get_parameters_conv(net.initial_stage, 'weight'), 'lr': base_lr},
        {'params': get_parameters_conv(net.initial_stage, 'bias'), 'lr': base_lr * 2, 'weight_decay': 0},
        {'params': get_parameters_conv(net.refinement_stages, 'weight'), 'lr': base_lr * 4},
        {'params': get_parameters_conv(net.refinement_stages, 'bias'), 'lr': base_lr * 8, 'weight_decay': 0},
        {'params': get_parameters_bn(net.refinement_stages, 'weight'), 'weight_decay': 0},
        {'params': get_parameters_bn(net.refinement_stages, 'bias'), 'lr': base_lr * 2, 'weight_decay': 0},
    ], lr=base_lr, weight_decay=5e-4)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200, 260], gamma=0.333)
    num_iter = 0
    current_epoch = 0

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        if from_mobilenet:
            load_from_mobilenet(net, checkpoint)
        else:
            load_state(net, checkpoint)
            if not weights_only:
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
                num_iter = checkpoint['iter']
                current_epoch = checkpoint['current_epoch']

    net = DataParallel(net).cuda()
    net.train()

    for epoch_id in range(current_epoch, 280):
        scheduler.step()
        total_losses = [0.0, 0.0] * (num_refinement_stages + 1)
        batch_iter_idx = 0

        for batch_data in train_loader:
            if batch_iter_idx == 0:
                optimizer.zero_grad()

            images        = batch_data['image'].cuda()
            kpt_masks     = batch_data['keypoint_mask'].cuda()
            paf_masks     = batch_data['paf_mask'].cuda()
            kpt_maps      = batch_data['keypoint_maps'].cuda()
            paf_maps      = batch_data['paf_maps'].cuda()
            stages_output = net(images)

            losses = []
            for loss_idx in range(len(total_losses) // 2):
                losses.append(l2_loss(stages_output[loss_idx * 2],     kpt_maps, kpt_masks, images.shape[0]))
                losses.append(l2_loss(stages_output[loss_idx * 2 + 1], paf_maps, paf_masks, images.shape[0]))
                total_losses[loss_idx * 2]     += losses[-2].item() / batches_per_iter
                total_losses[loss_idx * 2 + 1] += losses[-1].item() / batches_per_iter

            loss = sum(losses) / batches_per_iter
            loss.backward()
            batch_iter_idx += 1

            if batch_iter_idx == batches_per_iter:
                optimizer.step()
                batch_iter_idx = 0
                num_iter += 1
            else:
                continue

            if num_iter % log_after == 0:
                logger.info('Iter: %d', num_iter)
                for i in range(len(total_losses) // 2):
                    logger.info('stage%d_heatmaps_loss: %.4f  paf_loss: %.4f',
                                i + 1,
                                total_losses[i * 2] / log_after,
                                total_losses[i * 2 + 1] / log_after)
                total_losses = [0.0] * len(total_losses)

            if num_iter % checkpoint_after == 0:
                os.makedirs(checkpoints_folder, exist_ok=True)
                snap = os.path.join(checkpoints_folder, f'checkpoint_iter_{num_iter}.pth')
                torch.save({
                    'state_dict': net.module.state_dict(),
                    'optimizer':  optimizer.state_dict(),
                    'scheduler':  scheduler.state_dict(),
                    'iter':       num_iter,
                    'current_epoch': epoch_id,
                }, snap)
                logger.info('已保存检查点: %s', snap)

            if num_iter % val_after == 0:
                logger.info('开始验证...')
                evaluate(val_labels, val_output_name, val_images_folder, net)
                net.train()


def main():
    parser = argparse.ArgumentParser(description='OpenPose 轻量化模型训练')
    parser.add_argument('--prepared-train-labels', default=TRAIN_LABELS_PATH)
    parser.add_argument('--train-images-folder',   default=TRAIN_IMAGES_FOLDER)
    parser.add_argument('--num-refinement-stages', type=int, default=TRAIN_REFINEMENT_STAGES)
    parser.add_argument('--base-lr',               type=float, default=TRAIN_BASE_LR)
    parser.add_argument('--batch-size',            type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument('--batches-per-iter',      type=int, default=1)
    parser.add_argument('--num-workers',           type=int, default=TRAIN_NUM_WORKERS)
    parser.add_argument('--checkpoint-path',       default=None)
    parser.add_argument('--from-mobilenet',        action='store_true')
    parser.add_argument('--weights-only',          action='store_false')
    parser.add_argument('--experiment-name',       default=OUTPUT_DIR)
    parser.add_argument('--log-after',             type=int, default=100)
    parser.add_argument('--val-labels',            default=VAL_LABELS_PATH)
    parser.add_argument('--val-images-folder',     default=VAL_IMAGES_FOLDER)
    parser.add_argument('--checkpoint-after',      type=int, default=5000)
    parser.add_argument('--val-after',             type=int, default=5000)
    args = parser.parse_args()

    checkpoints_folder = f'{args.experiment_name}_checkpoints'
    val_output = os.path.join(OUTPUT_DIR, 'detections.json')

    train(
        prepared_train_labels=args.prepared_train_labels,
        train_images_folder=args.train_images_folder,
        num_refinement_stages=args.num_refinement_stages,
        base_lr=args.base_lr,
        batch_size=args.batch_size,
        batches_per_iter=args.batches_per_iter,
        num_workers=args.num_workers,
        checkpoint_path=args.checkpoint_path,
        weights_only=args.weights_only,
        from_mobilenet=args.from_mobilenet,
        checkpoints_folder=checkpoints_folder,
        log_after=args.log_after,
        val_labels=args.val_labels,
        val_images_folder=args.val_images_folder,
        val_output_name=val_output,
        checkpoint_after=args.checkpoint_after,
        val_after=args.val_after,
    )


if __name__ == '__main__':
    main()
