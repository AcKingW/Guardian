"""
Pipeline 公共工具函数。
提供在图像上叠加中文文字等跨模块共用功能。
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import FONT_SIMSUN, FONT_DEFAULT_SIZE


def cv2_add_chinese_text(img: np.ndarray, text: str,
                         left: int, top: int,
                         color: tuple = (0, 255, 0),
                         font_size: int = FONT_DEFAULT_SIZE) -> np.ndarray:
    """
    在 OpenCV 图像上叠加中文文字（OpenCV 原生不支持中文）。

    将 BGR 图像转换为 PIL → 绘制文字 → 转回 BGR。

    Args:
        img:       OpenCV BGR 图像（numpy array）
        text:      要绘制的文字
        left:      文字左边界 x 坐标
        top:       文字上边界 y 坐标
        color:     文字颜色 (R, G, B)
        font_size: 字号

    Returns:
        绘制文字后的 BGR 图像
    """
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype(FONT_SIMSUN, font_size, encoding='utf-8')
    draw.text((left, top), text, color, font=font)
    return cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
