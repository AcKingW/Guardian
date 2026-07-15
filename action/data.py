"""
行为分类数据集（骨骼图像 → 跌倒/正常 标签）。
"""
import os
import cv2
import numpy as np
from torch.utils.data import Dataset


class PoseDataset(Dataset):
    """
    骨骼姿态图像数据集。

    目录结构::

        root/
        ├── train/
        │   ├── fall/      ← 标签 0（跌倒）
        │   └── normal/    ← 标签 1（正常）
        └── test/
            ├── fall/
            └── normal/

    Args:
        root:     数据集根目录
        is_train: True → 加载 train/，False → 加载 test/
    """

    LABEL_MAP = {'fall': 0, 'normal': 1}

    def __init__(self, root: str, is_train: bool = True):
        super().__init__()
        self.dataset = []
        sub_dir = "train" if is_train else "test"

        for tag in os.listdir(os.path.join(root, sub_dir)):
            label = self.LABEL_MAP.get(tag, -1)
            if label == -1:
                continue
            file_dir = os.path.join(root, sub_dir, tag)
            for img_file in os.listdir(file_dir):
                self.dataset.append((os.path.join(file_dir, img_file), label))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, item: int):
        img_path, label = self.dataset[item]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = img.reshape(-1).astype(np.float32) / 255.0
        one_hot = np.zeros(2, dtype=np.float32)
        one_hot[label] = 1.0
        return img, one_hot
