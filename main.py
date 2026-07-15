"""
Guardian — 智能安防监控系统
统一命令行入口，支持三种运行模式：

    python main.py --mode count  --video path/to/video.mp4    # 人流量统计
    python main.py --mode fall   --video path/to/video.mp4    # 跌倒检测
    python main.py --mode gui                                  # 启动监控管理界面
    python main.py --mode camera --mode fall                   # 实时摄像头检测
"""
import argparse
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger('guardian')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='guardian',
        description='Guardian — 智能安防监控系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 人流量统计（视频文件）
  python main.py --mode count --video data/sample.mp4

  # 跌倒检测（视频文件）
  python main.py --mode fall --video data/sample.mp4

  # 跌倒检测（摄像头，ID=0）
  python main.py --mode fall --camera 0

  # 启动 GUI 监控管理界面
  python main.py --mode gui

  # 图片目录检测
  python main.py --mode fall --images data/frames/
        """,
    )

    parser.add_argument(
        '--mode', required=True,
        choices=['count', 'fall', 'gui'],
        help='运行模式: count=人流量统计 | fall=跌倒检测 | gui=监控管理界面',
    )
    parser.add_argument('--video',  default='', help='视频文件路径')
    parser.add_argument('--camera', default='', help='摄像头 ID（整数字符串，如 0）')
    parser.add_argument('--images', default='', help='图片路径或目录')
    parser.add_argument('--output', default='', help='输出视频路径（仅 count 模式有效）')
    parser.add_argument('--name',   default='Guardian', help='监控点名称（叠加到画面）')
    parser.add_argument('--cpu',    action='store_true', help='强制 CPU 推理（不使用 GPU）')
    return parser.parse_args()


def run_count(args: argparse.Namespace):
    """人流量统计模式。"""
    from detector.people_counter import PeopleCounterYOLO, detect_video

    video_path = args.camera if args.camera else args.video
    if not video_path:
        logger.error('count 模式需要 --video 或 --camera 参数')
        sys.exit(1)

    logger.info('启动人流量统计 | 视频源: %s', video_path)
    detector = PeopleCounterYOLO()
    detect_video(detector, video_path, output_path=args.output)


def run_fall(args: argparse.Namespace):
    """跌倒检测模式。"""
    from detector.fall_detector import FallDetector

    video_src = args.camera if args.camera else args.video
    image_src = args.images

    if not video_src and not image_src:
        logger.error('fall 模式需要 --video、--camera 或 --images 参数')
        sys.exit(1)

    logger.info('启动跌倒检测 | 视频源: %s  图片: %s', video_src, image_src)
    detector = FallDetector(cpu=args.cpu)
    detector.detect(
        video_source=video_src,
        image_source=image_src,
        video_name=args.name,
    )


def run_gui():
    """监控管理 GUI 模式。"""
    from gui.monitor_app import launch
    logger.info('启动 Guardian 监控管理界面')
    launch()


def main():
    args = parse_args()

    dispatch = {
        'count': lambda: run_count(args),
        'fall':  lambda: run_fall(args),
        'gui':   run_gui,
    }
    dispatch[args.mode]()


if __name__ == '__main__':
    main()
