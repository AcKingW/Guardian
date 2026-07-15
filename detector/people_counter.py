"""
人流量统计检测器（基于 YOLOv3 + Keras）。

合并了原始 people_flow.py 和 yolo.py 中几乎完全重复的 YOLO 类，
提供人员计数专用接口（仅保留 class_id=0 即 person 类别）。
"""
import colorsys
import logging
import os
from timeit import default_timer as timer

import cv2
import numpy as np
from keras import backend as K
from keras.layers import Input
from keras.models import load_model
from keras.utils import multi_gpu_model
from PIL import Image, ImageFont, ImageDraw

from yolo.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo.utils import letterbox_image
from config import (
    YOLO_MODEL_PATH, YOLO_ANCHORS_PATH, YOLO_CLASSES_PATH,
    YOLO_SCORE_THRESH, YOLO_IOU_THRESH, YOLO_INPUT_SIZE, YOLO_GPU_NUM,
    FONT_MONO, FONT_CHINESE,
)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
logger = logging.getLogger(__name__)


class PeopleCounterYOLO:
    """
    基于 YOLOv3 的人员计数检测器。

    只保留 COCO class_id=0（person）的检测结果，
    并在画面中叠加人数统计信息。

    Args:
        model_path:    YOLO .h5 模型/权重路径
        anchors_path:  anchor 文本文件路径
        classes_path:  类别名称文本文件路径
        score:         置信度阈值
        iou:           NMS IOU 阈值
        input_size:    输入尺寸 (h, w)
        gpu_num:       GPU 数量
    """

    def __init__(self,
                 model_path: str = YOLO_MODEL_PATH,
                 anchors_path: str = YOLO_ANCHORS_PATH,
                 classes_path: str = YOLO_CLASSES_PATH,
                 score: float = YOLO_SCORE_THRESH,
                 iou: float = YOLO_IOU_THRESH,
                 input_size: tuple = YOLO_INPUT_SIZE,
                 gpu_num: int = YOLO_GPU_NUM):
        self.model_path = os.path.expanduser(model_path)
        self.anchors_path = os.path.expanduser(anchors_path)
        self.classes_path = os.path.expanduser(classes_path)
        self.score = score
        self.iou = iou
        self.model_image_size = input_size
        self.gpu_num = gpu_num

        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self._generate()

    def _get_class(self) -> list:
        with open(self.classes_path) as f:
            return [c.strip() for c in f.readlines()]

    def _get_anchors(self) -> np.ndarray:
        with open(self.anchors_path) as f:
            anchors = [float(x) for x in f.readline().split(',')]
        return np.array(anchors).reshape(-1, 2)

    def _generate(self):
        """加载模型并生成输出张量。"""
        assert self.model_path.endswith('.h5'), '权重文件必须是 .h5 格式'
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny = (num_anchors == 6)

        try:
            self.yolo_model = load_model(self.model_path, compile=False)
        except Exception:
            body_fn = tiny_yolo_body if is_tiny else yolo_body
            anchor_per = num_anchors // 2 if is_tiny else num_anchors // 3
            self.yolo_model = body_fn(
                Input(shape=(None, None, 3)), anchor_per, num_classes
            )
            self.yolo_model.load_weights(self.model_path)
        else:
            expected = num_anchors / len(self.yolo_model.output) * (num_classes + 5)
            assert self.yolo_model.layers[-1].output_shape[-1] == expected, \
                '模型输出形状与 anchor/类别数不匹配'

        logger.info('已加载模型: %s', self.model_path)

        # 生成检测框颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in self.colors]
        np.random.seed(10101)
        np.random.shuffle(self.colors)
        np.random.seed(None)

        self.input_image_shape = K.placeholder(shape=(2,))
        if self.gpu_num >= 2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)

        return yolo_eval(
            self.yolo_model.output, self.anchors,
            len(self.class_names), self.input_image_shape,
            score_threshold=self.score, iou_threshold=self.iou,
        )

    def detect_image(self, image: Image.Image) -> Image.Image:
        """
        对单张 PIL 图像执行人员检测并绘制结果。

        Args:
            image: PIL Image（RGB）

        Returns:
            绘制了检测框和人数统计的 PIL Image
        """
        start = timer()

        # 预处理
        if self.model_image_size != (None, None):
            assert all(s % 32 == 0 for s in self.model_image_size), '输入尺寸必须是 32 的倍数'
            boxed = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_size = (
                image.width  - (image.width  % 32),
                image.height - (image.height % 32),
            )
            boxed = letterbox_image(image, new_size)

        image_data = np.expand_dims(np.array(boxed, dtype='float32') / 255.0, 0)

        # 推理
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0,
            },
        )

        # 只保留 person 类别（class_id = 0）
        person_mask = out_classes == 0
        out_boxes   = out_boxes[person_mask]
        out_scores  = out_scores[person_mask]
        out_classes = out_classes[person_mask]
        person_count = len(out_boxes)
        logger.debug('画面中有 %d 人', person_count)

        # 字体
        font_size = max(1, int(3e-2 * image.size[1] + 0.5))
        font    = ImageFont.truetype(font=FONT_MONO,    size=font_size)
        font_cn = ImageFont.truetype(font=FONT_CHINESE, size=font_size)
        thickness = max(1, (image.size[0] + image.size[1]) // 300)

        draw = ImageDraw.Draw(image)

        # 绘制检测框
        for i, c in reversed(list(enumerate(out_classes))):
            box   = out_boxes[i]
            score = out_scores[i]
            label = f'{self.class_names[c]} {score:.2f}'

            top    = max(0,              int(np.floor(box[0] + 0.5)))
            left   = max(0,              int(np.floor(box[1] + 0.5)))
            bottom = min(image.size[1],  int(np.floor(box[2] + 0.5)))
            right  = min(image.size[0],  int(np.floor(box[3] + 0.5)))

            label_size = draw.textsize(label, font)
            text_origin = np.array([left, top - label_size[1]]) \
                if top - label_size[1] >= 0 else np.array([left, top + 1])

            for t in range(thickness):
                draw.rectangle([left + t, top + t, right - t, bottom - t],
                                outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)],
                            fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)

        # 叠加人数统计
        count_text = f'  画面中有{person_count}人  '
        label_size = draw.textsize(count_text, font_cn)
        draw.rectangle([10, 10, 10 + label_size[0], 10 + label_size[1]], fill=(255, 255, 0))
        draw.text((10, 10), count_text, fill=(0, 0, 0), font=font_cn)

        del draw
        logger.debug('检测耗时: %.3fs', timer() - start)
        return image

    def close_session(self):
        """关闭 TensorFlow session，释放资源。"""
        self.sess.close()


# ─────────────────────────────────────────────
# 视频检测入口
# ─────────────────────────────────────────────

def detect_video(detector: PeopleCounterYOLO, video_path: str,
                 output_path: str = '') -> None:
    """
    对视频文件执行逐帧人员计数，可选保存输出视频。

    Args:
        detector:    已初始化的 PeopleCounterYOLO 实例
        video_path:  输入视频路径（或摄像头 ID 字符串）
        output_path: 输出视频路径，为空则不保存
    """
    try:
        source = int(video_path)
    except ValueError:
        source = video_path

    vid = cv2.VideoCapture(source)
    if not vid.isOpened():
        raise IOError(f'无法打开视频源: {video_path}')

    video_fourcc = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps    = vid.get(cv2.CAP_PROP_FPS)
    video_size   = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    out = None
    if output_path:
        out = cv2.VideoWriter(output_path, video_fourcc, video_fps, video_size)

    accum_time = 0.0
    curr_fps = 0
    fps_label = 'FPS: ??'
    prev_time = timer()

    try:
        while True:
            ret, frame = vid.read()
            if not ret:
                break

            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image = detector.detect_image(image)
            result = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

            curr_time = timer()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time += exec_time
            curr_fps += 1
            if accum_time > 1:
                accum_time -= 1
                fps_label = f'FPS: {curr_fps}'
                curr_fps = 0

            cv2.putText(result, fps_label, (3, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.namedWindow('Guardian - People Counter', cv2.WINDOW_NORMAL)
            cv2.imshow('Guardian - People Counter', result)
            if out:
                out.write(result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        vid.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        detector.close_session()
