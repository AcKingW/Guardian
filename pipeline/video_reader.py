"""
视频和图像源读取器（迭代器模式）。
支持视频文件、摄像头 ID 以及图片目录/单张图片。
"""
import os
import cv2
import numpy as np


class ImageReader:
    """
    图片列表迭代器。

    Args:
        file_names: 图片文件路径列表
    """

    def __init__(self, file_names: list):
        self.file_names = file_names
        self.max_idx = len(file_names)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self) -> np.ndarray:
        if self.idx == self.max_idx:
            raise StopIteration
        img = cv2.imread(self.file_names[self.idx], cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            raise IOError(f'无法读取图片: {self.file_names[self.idx]}')
        self.idx += 1
        return img


class VideoReader:
    """
    视频文件 / 摄像头迭代器，逐帧在左上角叠加摄像头名称。

    Args:
        file_name:  视频文件路径，或摄像头整数 ID（字符串形式也可）
        code_name:  叠加到画面的摄像头标识文字
    """

    def __init__(self, file_name: str, code_name: str = ''):
        self.code_name = str(code_name)
        try:
            self.file_name = int(file_name)  # 摄像头 ID
        except (ValueError, TypeError):
            self.file_name = file_name

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError(f'无法打开视频源: {self.file_name}')
        return self

    def __next__(self) -> np.ndarray:
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        if self.code_name:
            cv2.putText(img, self.code_name, (5, 35),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
        return img

    def __del__(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()


def build_frame_provider(video_path: str = '', image_path: str = '',
                         code_name: str = ''):
    """
    根据输入类型构建合适的帧提供者。

    Args:
        video_path:  视频文件路径或摄像头 ID（字符串）
        image_path:  图片文件或目录路径
        code_name:   视频流标签

    Returns:
        可迭代的帧提供者

    Raises:
        ValueError: 两个路径均为空时抛出
    """
    if video_path:
        return VideoReader(video_path, code_name)

    if image_path:
        if os.path.isdir(image_path):
            files = [
                os.path.join(image_path, f)
                for f in os.listdir(image_path)
            ]
            return ImageReader(files)
        else:
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            return [img]

    raise ValueError('video_path 和 image_path 至少需要提供一个。')
