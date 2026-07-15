"""
YOLOv3 模型训练脚本（Keras 实现）。
使用方式:
    python -m train.train_yolo
"""
import argparse
import logging
import os

import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from yolo.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo.utils import get_random_data
from config import (
    YOLO_ANCHORS_PATH, YOLO_VOC_CLASSES_PATH,
    YOLO_TRAIN_ANNOTATION, YOLO_TRAIN_LOG_DIR,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def get_classes(classes_path: str) -> list:
    """从文本文件加载类别名称列表。"""
    with open(classes_path) as f:
        return [c.strip() for c in f.readlines()]


def get_anchors(anchors_path: str) -> np.ndarray:
    """从文本文件加载 YOLO anchors。"""
    with open(anchors_path) as f:
        anchors = [float(x) for x in f.readline().split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, load_pretrained=True,
                 freeze_body=2, weights_path='model_data/yolo_weights.h5'):
    """构建 YOLOv3 训练模型。"""
    K.clear_session()
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [
        Input(shape=(h // {0: 32, 1: 16, 2: 8}[l],
                     w // {0: 32, 1: 16, 2: 8}[l],
                     num_anchors // 3, num_classes + 5))
        for l in range(3)
    ]

    model_body = yolo_body(image_input, num_anchors // 3, num_classes)
    logger.info('创建 YOLOv3，%d anchors，%d 类别', num_anchors, num_classes)

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        logger.info('已加载权重: %s', weights_path)
        if freeze_body in [1, 2]:
            num = (185, len(model_body.layers) - 3)[freeze_body - 1]
            for i in range(num):
                model_body.layers[i].trainable = False
            logger.info('冻结前 %d / %d 层', num, len(model_body.layers))

    model_loss = Lambda(
        yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5}
    )([*model_body.output, *y_true])
    return Model([model_body.input, *y_true], model_loss)


def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True,
                      freeze_body=2, weights_path='model_data/tiny_yolo_weights.h5'):
    """构建 Tiny YOLOv3 训练模型。"""
    K.clear_session()
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [
        Input(shape=(h // {0: 32, 1: 16}[l],
                     w // {0: 32, 1: 16}[l],
                     num_anchors // 2, num_classes + 5))
        for l in range(2)
    ]

    model_body = tiny_yolo_body(image_input, num_anchors // 2, num_classes)
    logger.info('创建 Tiny YOLOv3，%d anchors，%d 类别', num_anchors, num_classes)

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        if freeze_body in [1, 2]:
            num = (20, len(model_body.layers) - 2)[freeze_body - 1]
            for i in range(num):
                model_body.layers[i].trainable = False

    model_loss = Lambda(
        yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7}
    )([*model_body.output, *y_true])
    return Model([model_body.input, *y_true], model_loss)


def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    """YOLOv3 数据生成器（无限循环，用于 fit_generator）。"""
    n = len(annotation_lines)
    i = 0
    while True:
        image_data, box_data = [], []
        for _ in range(batch_size):
            if i == 0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i + 1) % n
        image_data = np.array(image_data)
        box_data   = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)


def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    """数据生成器包装（校验空数据）。"""
    if len(annotation_lines) == 0 or batch_size <= 0:
        return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)


def train(annotation_path: str = YOLO_TRAIN_ANNOTATION,
          log_dir: str = YOLO_TRAIN_LOG_DIR,
          classes_path: str = YOLO_VOC_CLASSES_PATH,
          anchors_path: str = YOLO_ANCHORS_PATH):
    """执行 YOLOv3 两阶段训练（冻结层 → 全量微调）。"""
    os.makedirs(log_dir, exist_ok=True)
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors     = get_anchors(anchors_path)
    input_shape = (416, 416)

    is_tiny_version = (len(anchors) == 6)
    if is_tiny_version:
        model = create_tiny_model(input_shape, anchors, num_classes,
                                  freeze_body=2,
                                  weights_path='model_data/tiny_yolo_weights.h5')
    else:
        model = create_model(input_shape, anchors, num_classes,
                             freeze_body=2,
                             weights_path='model_data/yolo_weights.h5')

    tb_logging   = TensorBoard(log_dir=log_dir)
    checkpoint   = ModelCheckpoint(
        os.path.join(log_dir, 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'),
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=3,
    )
    reduce_lr    = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stop   = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val   = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    # ── 阶段一：冻结骨干，快速收敛 ──────────────
    batch_size = 32
    model.compile(
        optimizer=Adam(lr=1e-3),
        loss={'yolo_loss': lambda y_true, y_pred: y_pred},
    )
    logger.info('阶段一 | train=%d  val=%d  batch=%d', num_train, num_val, batch_size)
    model.fit_generator(
        data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
        steps_per_epoch=max(1, num_train // batch_size),
        validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
        validation_steps=max(1, num_val // batch_size),
        epochs=50, initial_epoch=0,
        callbacks=[tb_logging, checkpoint],
    )
    model.save_weights(os.path.join(log_dir, 'trained_weights_stage_1.h5'))

    # ── 阶段二：解冻全网络，精细微调 ──────────────
    for layer in model.layers:
        layer.trainable = True
    model.compile(
        optimizer=Adam(lr=1e-4),
        loss={'yolo_loss': lambda y_true, y_pred: y_pred},
    )
    logger.info('阶段二（全量微调）| train=%d  val=%d  batch=%d', num_train, num_val, batch_size)
    model.fit_generator(
        data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
        steps_per_epoch=max(1, num_train // batch_size),
        validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
        validation_steps=max(1, num_val // batch_size),
        epochs=100, initial_epoch=50,
        callbacks=[tb_logging, checkpoint, reduce_lr, early_stop],
    )
    model.save_weights(os.path.join(log_dir, 'trained_weights_final.h5'))
    logger.info('训练完成，权重已保存至 %s', log_dir)


def main():
    parser = argparse.ArgumentParser(description='YOLOv3 人流量检测模型训练')
    parser.add_argument('--annotation', default=YOLO_TRAIN_ANNOTATION, help='标注文件路径')
    parser.add_argument('--log-dir',    default=YOLO_TRAIN_LOG_DIR,    help='日志与权重输出目录')
    parser.add_argument('--classes',    default=YOLO_VOC_CLASSES_PATH, help='类别文件路径')
    parser.add_argument('--anchors',    default=YOLO_ANCHORS_PATH,     help='Anchor 文件路径')
    args = parser.parse_args()
    train(args.annotation, args.log_dir, args.classes, args.anchors)


if __name__ == '__main__':
    main()
