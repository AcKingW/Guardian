"""Miscellaneous utility functions for YOLO data processing."""

from functools import reduce

import numpy as np
from PIL import Image
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb


def compose(*funcs):
    """
    从左到右组合任意数量的函数。

    Example:
        compose(f, g)(x) == g(f(x))
    """
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    raise ValueError('不支持对空序列进行函数组合。')


def letterbox_image(image: Image.Image, size: tuple) -> Image.Image:
    """
    保持原始宽高比缩放图像至目标尺寸，不足部分用灰色填充。

    Args:
        image: PIL Image
        size:  目标尺寸 (w, h)

    Returns:
        填充后的 PIL Image
    """
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image


def rand(a: float = 0, b: float = 1) -> float:
    """在 [a, b) 内生成均匀分布随机数。"""
    return np.random.rand() * (b - a) + a


def get_random_data(annotation_line: str, input_shape: tuple,
                    random: bool = True, max_boxes: int = 20,
                    jitter: float = 0.3, hue: float = 0.1,
                    sat: float = 1.5, val: float = 1.5,
                    proc_img: bool = True):
    """
    实时数据增强预处理（训练用）。

    Args:
        annotation_line: VOC 格式标注行 "img_path x1,y1,x2,y2,cls ..."
        input_shape:     网络输入形状 (h, w)
        random:          True → 随机增强，False → 仅 resize
        max_boxes:       单图最大框数
        jitter:          宽高比抖动范围
        hue/sat/val:     色调/饱和度/亮度增强幅度
        proc_img:        是否处理图像（False 仅处理框）

    Returns:
        (image_data, box_data): (H, W, 3) float32，(max_boxes, 5) int
    """
    line = annotation_line.split()
    image = Image.open(line[0])
    iw, ih = image.size
    h, w = input_shape
    box = np.array([np.array(list(map(int, b.split(',')))) for b in line[1:]])

    if not random:
        scale = min(w / iw, h / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        dx, dy = (w - nw) // 2, (h - nh) // 2
        image_data = 0
        if proc_img:
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image) / 255.0

        box_data = np.zeros((max_boxes, 5))
        if len(box) > 0:
            np.random.shuffle(box)
            box = box[:max_boxes]
            box[:, [0, 2]] = box[:, [0, 2]] * scale + dx
            box[:, [1, 3]] = box[:, [1, 3]] * scale + dy
            box_data[:len(box)] = box
        return image_data, box_data

    # 随机缩放与位置
    new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
    scale = rand(0.25, 2)
    if new_ar < 1:
        nh, nw = int(scale * h), int(scale * h * new_ar)
    else:
        nw, nh = int(scale * w), int(scale * w / new_ar)

    image = image.resize((nw, nh), Image.BICUBIC)
    dx, dy = int(rand(0, w - nw)), int(rand(0, h - nh))
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image = new_image

    if rand() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # 色彩增强
    hue_delta = rand(-hue, hue)
    sat_factor = rand(1, sat) if rand() < 0.5 else 1 / rand(1, sat)
    val_factor = rand(1, val) if rand() < 0.5 else 1 / rand(1, val)
    x = rgb_to_hsv(np.array(image) / 255.0)
    x[..., 0] = (x[..., 0] + hue_delta) % 1.0
    x[..., 1] = np.clip(x[..., 1] * sat_factor, 0, 1)
    x[..., 2] = np.clip(x[..., 2] * val_factor, 0, 1)
    image_data = hsv_to_rgb(x)

    box_data = np.zeros((max_boxes, 5))
    if len(box) > 0:
        np.random.shuffle(box)
        box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
        box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w > 1, box_h > 1)]
        box = box[:max_boxes]
        box_data[:len(box)] = box

    return image_data, box_data
