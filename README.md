# Guardian — 智守护-智能行车安全系统

> 基于 **OpenPose + YOLOv3** 的综合安防监控平台，集成跌倒/跳跃行为检测、人流量统计和移动端管理 App。

---

## 项目结构

```
Guardian/
├── main.py                 # 统一命令行入口
├── config.py               # 全局配置（路径、阈值等）
├── requirements.txt        # Python 依赖
│
├── detector/               # 检测器接口层
│   ├── people_counter.py   # 人流量统计（YOLOv3）
│   └── fall_detector.py    # 跌倒/跳跃检测（OpenPose）
│
├── pose/                   # 姿态估计核心库（OpenPose）
│   ├── keypoints.py        # 关键点提取与组合
│   ├── pose.py             # Pose 类（关键点/骨骼/角度）
│   ├── one_euro_filter.py  # 平滑滤波器
│   ├── conv.py             # 卷积模块工厂
│   ├── get_parameters.py   # 优化器参数分组
│   ├── load_state.py       # 权重加载工具
│   └── loss.py             # L2 损失函数
│
├── action/                 # 行为分类模块
│   ├── net.py              # 网络架构（NetV2 等）
│   ├── data.py             # PoseDataset 数据集
│   ├── detect.py           # 推理接口
│   └── train.py            # ActionTrainer 训练器
│
├── yolo/                   # YOLOv3 目标检测库（Keras）
│   ├── model.py            # 网络定义
│   └── utils.py            # 数据增强工具
│
├── pipeline/               # 视频处理管线
│   ├── video_reader.py     # 帧提供者（视频/图片/摄像头）
│   ├── inference.py        # OpenPose 推理与可视化
│   └── utils.py            # 公共工具（中文字体叠加等）
│
├── gui/                    # 桌面管理界面
│   └── monitor_app.py      # Tkinter 监控管理 App
│
├── train/                  # 训练脚本
│   ├── train_pose.py       # 训练 OpenPose 姿态估计
│   ├── train_yolo.py       # 训练 YOLOv3 人员检测
│   └── val_pose.py         # COCO 验证集评估
│
├── misc/                   # 独立工具脚本
│   └── sd.py               # 樱花动画 + 烟花粒子效果演示
│
├── app/                    # HyBrid 移动端前端（uni-app）
│   └── chebiaoton/         # 车标通 App 源码
│
├── weights/                # 模型权重目录（不入 git）
│   ├── openpose.jit
│   └── action.jit
│
├── model_data/             # YOLO 模型配置
│   ├── yolo.h5
│   ├── yolo_anchors.txt
│   └── coco_classes.txt
│
├── font/                   # 字体文件
│   ├── FiraMono-Medium.otf
│   ├── asl.otf
│   └── simsun.ttc
│
└── Video_Information/      # 监控视频源配置
    ├── url_address.txt
    └── id_name.txt
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备权重文件

将以下文件放入对应目录：

| 文件 | 目录 | 用途 |
|------|------|------|
| `openpose.jit` | `weights/` | OpenPose 姿态估计 |
| `action.jit` | `weights/` | 行为分类（跌倒检测） |
| `yolo.h5` | `model_data/` | YOLOv3 人员检测 |
| `yolo_anchors.txt` | `model_data/` | YOLO anchors |
| `coco_classes.txt` | `model_data/` | COCO 80 类别名称 |
| `FiraMono-Medium.otf` | `font/` | 标注字体 |
| `asl.otf` | `font/` | 中文标注字体 |
| `simsun.ttc` | `font/` | 中文叠加字体 |

### 3. 运行

```bash
# 人流量统计（视频文件）
python main.py --mode count --video path/to/video.mp4

# 跌倒/跳跃行为检测（视频文件）
python main.py --mode fall --video path/to/video.mp4

# 实时摄像头检测（摄像头 ID=0）
python main.py --mode fall --camera 0

# 启动 GUI 监控管理界面
python main.py --mode gui

# CPU 模式（无 GPU）
python main.py --mode fall --video video.mp4 --cpu
```

---

## 功能模块

### 🔍 跌倒 / 跳跃检测（`--mode fall`）

- 基于 **OpenPose 轻量化模型**提取 18 个人体关键点
- 通过**宽高比融合分类网络**判定跌倒（`fallPeople` / `normalPeople`）
- 通过**帧间颈部高度差**检测跳跃行为
- 支持多路监控视频并发检测（多线程）

### 👥 人流量统计（`--mode count`）

- 基于 **YOLOv3（Keras）**检测画面中的所有人员
- 实时在画面上叠加当前帧人数
- 可将检测结果保存为视频文件

### 🖥️ GUI 监控管理（`--mode gui`）

- Tkinter 桌面界面，管理监控视频源列表
- 支持动态添加/删除摄像头地址
- 一键启动所有监控点并发检测

### 📱 移动端 App（`app/chebiaoton/`）

- uni-app / HyBrid 前端，支持 Android/iOS
- 功能：登录注册、地图定位、报修上报、聊天、OBD 诊断等
- 独立部署，通过 API 与后端通信

---

## 训练自定义模型

### 训练 OpenPose 姿态估计

```bash
python -m train.train_pose \
  --train-images-folder /path/to/coco/train2017 \
  --prepared-train-labels data/prepared_train_annotation.pkl
```

### 训练行为分类网络

```bash
python -m action.train --data-root /path/to/pose_dataset
```

数据集目录结构：
```
dataset/
├── train/
│   ├── fall/    ← 跌倒骨骼图（128×128 灰度）
│   └── normal/  ← 正常骨骼图
└── test/
    ├── fall/
    └── normal/
```

### 训练 YOLOv3 人员检测

```bash
python -m train.train_yolo \
  --annotation data/train.txt \
  --classes model_data/voc_classes.txt
```

---

## 配置说明

修改 [`config.py`](config.py) 调整各项参数：

```python
# 检测阈值
YOLO_SCORE_THRESH  = 0.3    # 人员检测置信度阈值
FALL_PROBABILITY_THRESHOLD = 0.55  # 跌倒判定概率阈值
JUMP_HEIGHT_DIFF_THRESHOLD = 38    # 跳跃高度差阈值（像素）

# 模型路径
POSE_CHECKPOINT_PATH   = "weights/openpose.jit"
ACTION_CHECKPOINT_PATH = "weights/action.jit"
YOLO_MODEL_PATH        = "model_data/yolo.h5"
```

---

## 技术栈

| 模块 | 技术 |
|------|------|
| 姿态估计 | OpenPose 轻量化（PyTorch JIT） |
| 人员检测 | YOLOv3（Keras / TensorFlow） |
| 行为分类 | 全连接网络（PyTorch） |
| 图像处理 | OpenCV、Pillow |
| GUI | Tkinter |
| 移动端 | uni-app（HyBrid） |

---

## License

本项目整合了多个开源项目的代码，请参阅各模块的原始许可证。
