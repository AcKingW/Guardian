"""
Guardian 全局配置
将原来各文件中的硬编码路径、阈值等统一管理于此，
修改此文件即可调整整个项目行为，无需逐个文件修改。
"""
import os

# ─────────────────────────────────────────────
# 基础路径：以项目根目录为起点
# ─────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────
# YOLO（人流量统计）配置
# ─────────────────────────────────────────────
YOLO_MODEL_PATH    = os.path.join(ROOT_DIR, "model_data", "yolo.h5")
YOLO_ANCHORS_PATH  = os.path.join(ROOT_DIR, "model_data", "yolo_anchors.txt")
YOLO_CLASSES_PATH  = os.path.join(ROOT_DIR, "model_data", "coco_classes.txt")
YOLO_SCORE_THRESH  = 0.3       # 检测置信度阈值
YOLO_IOU_THRESH    = 0.45      # NMS IOU 阈值
YOLO_INPUT_SIZE    = (416, 416) # 模型输入尺寸 (h, w)，必须是 32 的倍数
YOLO_GPU_NUM       = 1          # 使用 GPU 数量（≥2 时启用多 GPU）

# ─────────────────────────────────────────────
# OpenPose（姿态估计 / 跌倒检测）配置
# ─────────────────────────────────────────────
POSE_CHECKPOINT_PATH  = os.path.join(ROOT_DIR, "weights", "openpose.jit")
ACTION_CHECKPOINT_PATH = os.path.join(ROOT_DIR, "weights", "action.jit")
POSE_HEIGHT_SIZE      = 256     # 网络输入层高度
POSE_STRIDE           = 8
POSE_UPSAMPLE_RATIO   = 4
POSE_CPU_MODE         = False   # True → 强制使用 CPU 推理

# ─────────────────────────────────────────────
# 行为分类阈值
# ─────────────────────────────────────────────
FALL_PROBABILITY_THRESHOLD = 0.55  # 跌倒概率阈值（超过则判定为跌倒）

# 跳跃检测参数
JUMP_HEIGHT_DIFF_THRESHOLD = 38   # 帧间高度差阈值（像素），超过则判定为跳跃
JUMP_DISPLAY_DURATION      = 10   # 跳跃/跌倒字幕显示帧数

# ─────────────────────────────────────────────
# 字体路径
# ─────────────────────────────────────────────
FONT_DIR          = os.path.join(ROOT_DIR, "font")
FONT_MONO         = os.path.join(FONT_DIR, "FiraMono-Medium.otf")
FONT_CHINESE      = os.path.join(FONT_DIR, "asl.otf")
FONT_SIMSUN       = os.path.join(FONT_DIR, "simsun.ttc")
FONT_DEFAULT_SIZE = 20          # 中文文字默认字号

# ─────────────────────────────────────────────
# 监控视频源配置
# ─────────────────────────────────────────────
VIDEO_INFO_DIR   = os.path.join(ROOT_DIR, "Video_Information")
URL_ADDRESS_FILE = os.path.join(VIDEO_INFO_DIR, "url_address.txt")
ID_NAME_FILE     = os.path.join(VIDEO_INFO_DIR, "id_name.txt")

# ─────────────────────────────────────────────
# 训练配置（OpenPose）
# ─────────────────────────────────────────────
TRAIN_LABELS_PATH    = os.path.join(ROOT_DIR, "data", "prepared_train_annotation.pkl")
TRAIN_IMAGES_FOLDER  = ""      # COCO train2017 路径（请根据本地环境修改）
VAL_LABELS_PATH      = os.path.join(ROOT_DIR, "data", "val_subset.json")
VAL_IMAGES_FOLDER    = ""      # COCO val2017 路径（请根据本地环境修改）
OUTPUT_DIR           = os.path.join(ROOT_DIR, "output")
CHECKPOINTS_DIR      = os.path.join(OUTPUT_DIR, "checkpoints")

# 训练超参数
TRAIN_BASE_LR        = 4e-5
TRAIN_BATCH_SIZE     = 6
TRAIN_NUM_WORKERS    = 1
TRAIN_REFINEMENT_STAGES = 3

# ─────────────────────────────────────────────
# 行为分类训练配置
# ─────────────────────────────────────────────
ACTION_DATA_ROOT       = ""     # 行为数据集根目录（请根据本地环境修改）
ACTION_CHECKPOINT_SAVE = os.path.join(ROOT_DIR, "weights", "action.pt")
ACTION_LOG_DIR         = os.path.join(ROOT_DIR, "logs")
ACTION_DEVICE          = "cuda:0"

# ─────────────────────────────────────────────
# YOLO 训练配置
# ─────────────────────────────────────────────
YOLO_TRAIN_ANNOTATION  = os.path.join(ROOT_DIR, "data", "train.txt")
YOLO_TRAIN_LOG_DIR     = os.path.join(ROOT_DIR, "logs", "yolo")
YOLO_VOC_CLASSES_PATH  = os.path.join(ROOT_DIR, "model_data", "voc_classes.txt")
