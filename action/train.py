"""
行为分类模型训练脚本。
使用方式:
    python -m action.train --data-root /path/to/dataset
"""
import argparse
import logging
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from action.data import PoseDataset
from action.net import NetV2
from config import ACTION_CHECKPOINT_SAVE, ACTION_LOG_DIR, ACTION_DEVICE, ACTION_DATA_ROOT

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


class ActionTrainer:
    """
    行为分类网络训练器。

    Args:
        data_root:   数据集根目录
        device:      训练设备（'cuda:0' / 'cpu'）
        checkpoint:  预训练权重路径（可选）
        log_dir:     TensorBoard 日志目录
    """

    def __init__(self, data_root: str, device: str = ACTION_DEVICE,
                 checkpoint: str = None, log_dir: str = ACTION_LOG_DIR):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.writer = SummaryWriter(log_dir)

        train_ds = PoseDataset(data_root, is_train=True)
        test_ds  = PoseDataset(data_root, is_train=False)
        self.train_loader = DataLoader(train_ds, batch_size=100, shuffle=True)
        self.test_loader  = DataLoader(test_ds,  batch_size=100, shuffle=True)

        self.net = NetV2().to(self.device)
        if checkpoint:
            self.net.load_state_dict(torch.load(checkpoint, map_location=self.device))
            logger.info("已加载权重: %s", checkpoint)

        self.optimizer = optim.Adam(self.net.parameters())

    def train(self, epochs: int = 100_000, save_path: str = ACTION_CHECKPOINT_SAVE):
        """执行训练循环。"""
        for epoch in range(epochs):
            # ── 训练阶段 ──────────────────────────────
            self.net.train()
            train_loss = 0.0
            for imgs, tags in self.train_loader:
                imgs, tags = imgs.to(self.device), tags.to(self.device)
                pred = self.net(imgs)
                loss = torch.mean((tags - pred) ** 2)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            train_loss /= len(self.train_loader)

            # ── 验证阶段 ──────────────────────────────
            self.net.eval()
            test_loss = 0.0
            with torch.no_grad():
                for imgs, tags in self.test_loader:
                    imgs, tags = imgs.to(self.device), tags.to(self.device)
                    pred = self.net(imgs)
                    test_loss += torch.mean((tags - pred) ** 2).item()
            test_loss /= len(self.test_loader)

            self.writer.add_scalars(
                "loss",
                {"train": train_loss, "test": test_loss},
                epoch,
            )
            logger.info("Epoch %d | train_loss=%.4f  test_loss=%.4f", epoch, train_loss, test_loss)
            torch.save(self.net.state_dict(), save_path)


def main():
    parser = argparse.ArgumentParser(description="行为分类模型训练")
    parser.add_argument('--data-root', type=str, default=ACTION_DATA_ROOT,
                        help='数据集根目录（含 train/ 和 test/ 子目录）')
    parser.add_argument('--device',    type=str, default=ACTION_DEVICE)
    parser.add_argument('--checkpoint', type=str, default=None, help='预训练权重路径')
    parser.add_argument('--epochs',    type=int, default=100_000)
    parser.add_argument('--save-path', type=str, default=ACTION_CHECKPOINT_SAVE)
    parser.add_argument('--log-dir',   type=str, default=ACTION_LOG_DIR)
    args = parser.parse_args()

    trainer = ActionTrainer(
        data_root=args.data_root,
        device=args.device,
        checkpoint=args.checkpoint,
        log_dir=args.log_dir,
    )
    trainer.train(epochs=args.epochs, save_path=args.save_path)


if __name__ == '__main__':
    main()
