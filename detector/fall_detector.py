"""
跌倒/跳跃行为检测器（基于 OpenPose + PyTorch）。
对外暴露简洁的 FallDetector 接口，封装模型加载与推理逻辑。
"""
import logging
import threading

from config import (
    POSE_CHECKPOINT_PATH, ACTION_CHECKPOINT_PATH, POSE_CPU_MODE,
    URL_ADDRESS_FILE, ID_NAME_FILE, VIDEO_INFO_DIR,
)
from pipeline.inference import load_and_run

logger = logging.getLogger(__name__)


class FallDetector:
    """
    跌倒 / 跳跃行为检测器。

    支持单视频源检测和多路监控视频并发检测。

    Args:
        pose_checkpoint:   OpenPose JIT 权重路径
        action_checkpoint: 行为分类 JIT 权重路径
        cpu:               True → 强制 CPU 推理
    """

    def __init__(self,
                 pose_checkpoint: str = POSE_CHECKPOINT_PATH,
                 action_checkpoint: str = ACTION_CHECKPOINT_PATH,
                 cpu: bool = POSE_CPU_MODE):
        self.pose_checkpoint = pose_checkpoint
        self.action_checkpoint = action_checkpoint
        self.cpu = cpu

    def detect(self, video_source: str = '', image_source: str = '',
               video_name: str = 'Guardian') -> None:
        """
        启动单路检测（阻塞，直到视频结束或按 'q' 退出）。

        Args:
            video_source: 视频文件路径 / 摄像头 ID
            image_source: 图片路径 / 目录
            video_name:   监控点名称
        """
        load_and_run(
            video_source=video_source,
            image_source=image_source,
            video_name=video_name,
            pose_checkpoint=self.pose_checkpoint,
            action_checkpoint=self.action_checkpoint,
            cpu=self.cpu,
        )

    def detect_multi(self, sources: list[dict]) -> list[threading.Thread]:
        """
        并发检测多路视频流（每路一个守护线程）。

        Args:
            sources: 字典列表，每项包含 'video_source' 和 'video_name' 键

        Returns:
            已启动的线程列表
        """
        threads = []
        for src in sources:
            t = threading.Thread(
                target=self.detect,
                kwargs=src,
                daemon=True,
            )
            t.start()
            threads.append(t)
            logger.info('已启动检测线程: %s', src.get('video_name', ''))
        return threads

    @staticmethod
    def load_sources_from_files(url_file: str = URL_ADDRESS_FILE,
                                name_file: str = ID_NAME_FILE) -> list[dict]:
        """
        从 Video_Information 目录下的配置文件读取视频源列表。

        Args:
            url_file:  url_address.txt 路径
            name_file: id_name.txt 路径

        Returns:
            字典列表 [{'video_source': ..., 'video_name': ...}, ...]
        """
        sources = []
        try:
            with open(url_file, 'r', encoding='utf-8') as uf, \
                 open(name_file, 'r', encoding='utf-8') as nf:
                for url, name in zip(uf, nf):
                    sources.append({
                        'video_source': url.strip(),
                        'video_name':   name.strip(),
                    })
        except FileNotFoundError as e:
            logger.warning('视频源配置文件不存在: %s', e)
        return sources
