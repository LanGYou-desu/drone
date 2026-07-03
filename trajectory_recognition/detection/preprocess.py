"""
视频预处理 — Video Preprocessing

基于 OpenCV 的视频读取与帧提取。
支持本地文件、RTSP/HTTP 流、USB 摄像头。
"""

from dataclasses import dataclass
from typing import Generator, Optional, Union

import numpy as np


@dataclass
class Frame:
    """预处理后的单帧数据"""
    image: np.ndarray        # BGR 图像 (H, W, 3)
    frame_id: int            # 帧序号（从 0 开始）
    timestamp: float         # 时间戳（秒）


class VideoProcessor:
    """
    视频预处理器，支持抽帧与尺寸标准化。

    使用示例:
        proc = VideoProcessor(frame_interval=5, target_size=(640, 640))
        for frame in proc.extract_frames("drone_video.mp4"):
            detections = detector.detect(frame.image)
    """

    def __init__(
        self,
        frame_interval: int = 5,
        target_size: tuple[int, int] = (640, 640),
        max_frames: Optional[int] = None,
    ):
        """
        Args:
            frame_interval: 抽帧间隔（每 N 帧处理一次，1 = 逐帧）
            target_size: 缩放目标尺寸 (width, height)，用于显示/预处理
            max_frames: 最大处理帧数，None 表示处理全部
        """
        self.frame_interval = frame_interval
        self.target_size = target_size
        self.max_frames = max_frames
        self._cap = None
        self._fps = 0.0
        self._total_frames = 0

    def extract_frames(
        self, source: Union[str, int]
    ) -> Generator[Frame, None, None]:
        """
        从视频源逐帧提取（抽帧）。

        Args:
            source: 视频源
                - str: 本地文件路径（mp4/avi/mov）或 RTSP/HTTP 流 URL
                - int: 摄像头设备 ID（0 = 默认摄像头，1 = 第二个...）

        Yields:
            Frame: 预处理后的帧数据（BGR 格式）

        Raises:
            FileNotFoundError: 文件不存在
            RuntimeError: 无法打开视频流
        """
        try:
            import cv2
        except ImportError:
            raise ImportError(
                "opencv-python 未安装，请运行: pip install opencv-python"
            )

        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(f"无法打开视频源: {source}")

        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        frame_count = 0
        extracted_count = 0

        while True:
            ret, image = self._cap.read()
            if not ret:
                break

            # 抽帧
            if frame_count % self.frame_interval != 0:
                frame_count += 1
                continue

            timestamp = frame_count / self._fps

            yield Frame(
                image=image,
                frame_id=frame_count,
                timestamp=round(timestamp, 3),
            )

            frame_count += 1
            extracted_count += 1

            if self.max_frames and extracted_count >= self.max_frames:
                break

        self._cap.release()
        self._cap = None

    def get_video_info(self, source: Union[str, int]) -> dict:
        """
        获取视频元信息（仅读取头，不遍历帧）。

        Returns:
            {
                "fps": 30.0,
                "total_frames": 900,
                "duration": 30.0,
                "width": 1920,
                "height": 1080,
            }
        """
        try:
            import cv2
        except ImportError:
            raise ImportError("opencv-python 未安装")

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频源: {source}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = total_frames / fps if fps > 0 else 0
        cap.release()

        return {
            "fps": round(fps, 2),
            "total_frames": total_frames,
            "duration": round(duration, 1),
            "width": width,
            "height": height,
        }

    def seek(self, frame_id: int) -> bool:
        """
        跳转到指定帧号（用于回溯分析）。

        Returns:
            是否成功跳转
        """
        if self._cap is None:
            return False
        self._cap.set(1, frame_id)  # cv2.CAP_PROP_POS_FRAMES = 1
        return True

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def total_frames(self) -> int:
        return self._total_frames
