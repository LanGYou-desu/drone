"""
YOLO 检测引擎 — YOLO Detection Engine

基于 ultralytics 封装 YOLO 模型的加载与推理。
支持 YOLOv8 / YOLOv11 系列模型，自动处理 GPU/CPU 设备切换。
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Detection:
    """单帧检测结果"""
    bbox: list[float]        # [x1, y1, x2, y2] 像素坐标（左上+右下）
    confidence: float         # 置信度 [0, 1]
    class_id: int             # 类别 ID
    class_name: str           # 类别名称（如 "drone", "bird"）


class YOLODetector:
    """
    YOLO 模型封装，统一推理接口。

    使用示例:
        detector = YOLODetector("models/yolov8n.pt", confidence=0.5, device="cuda:0")
        detections = detector.detect(frame)

    支持模型:
        - YOLOv8:  yolov8n.pt / yolov8s.pt / yolov8m.pt / yolov8l.pt / yolov8x.pt
        - YOLOv11: yolov11n.pt / yolov11s.pt / yolov11m.pt / yolov11l.pt / yolov11x.pt
        - 自定义: 任意 .pt 权重文件
    """

    # YOLO 模型的默认类别名（COCO 数据集）
    COCO_CLASSES = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush",
    ]

    # 无人机相关：COCO 中没有 "drone"，需要用自定义模型。
    # 默认关注 "bird"(14) 和 "airplane"(4) 作为空中目标代理，
    # 训练自定义 drone 模型后 class_id 0 即为 drone。

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.5,
        nms_threshold: float = 0.45,
        device: str = "cpu",
        input_size: tuple[int, int] = (640, 640),
        target_classes: Optional[list[int]] = None,
    ):
        """
        初始化 YOLO 检测器。

        Args:
            model_path: 模型权重文件路径（.pt）
            confidence: 置信度阈值，低于此值的检测框被过滤
            nms_threshold: NMS 交并比阈值
            device: 推理设备（"cpu" / "cuda:0" / "mps"）
            input_size: 模型输入尺寸 (width, height)，通常 640
            target_classes: 关注的类别 ID 列表，None 表示不过滤。
                           无人机场景建议设为 [4] (airplane) 或自定义 drone 模型后设为 [0]
        """
        self.model_path = model_path
        self.confidence = confidence
        self.nms_threshold = nms_threshold
        self.device = device
        self.input_size = input_size
        self.target_classes = target_classes
        self._model = None

    @property
    def model(self):
        """延迟加载模型（首次推理时加载）"""
        if self._model is None:
            self._load_model()
        return self._model

    def _load_model(self):
        """加载 YOLO 模型权重"""
        try:
            from ultralytics import YOLO
            # 自动检测 GPU
            device = self.device
            if device == "auto":
                try:
                    import torch
                    device = "0" if torch.cuda.is_available() else "cpu"
                except Exception:
                    device = "cpu"
                self.device = device
            self._model = YOLO(self.model_path)
            print(f"[YOLO] 模型已加载: {self.model_path} on {self.device}")
        except ImportError:
            raise ImportError(
                "ultralytics 未安装，请运行: pip install ultralytics"
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"模型文件不存在: {self.model_path}\n"
                "自动下载预训练模型:\n"
                "  from ultralytics import YOLO\n"
                "  model = YOLO('yolov8n.pt')  # 首次运行自动下载\n\n"
                "或训练自定义 drone 模型: python -m trajectory_recognition.train"
            )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        对单帧图像执行目标检测。

        Args:
            frame: BGR 图像 (H, W, 3)，numpy 数组（OpenCV 格式）

        Returns:
            检测结果列表，按置信度降序排列
        """
        # ultralytics 内部处理 RGB/BGR 转换
        results = self.model(
            frame,
            conf=self.confidence,
            iou=self.nms_threshold,
            imgsz=self.input_size,
            device=self.device,
            verbose=False,
        )

        detections = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())

                # 类别过滤
                if self.target_classes is not None and cls_id not in self.target_classes:
                    continue

                xyxy = boxes.xyxy[i].tolist()  # [x1, y1, x2, y2]
                cls_name = (
                    self.COCO_CLASSES[cls_id]
                    if cls_id < len(self.COCO_CLASSES)
                    else f"class_{cls_id}"
                )

                detections.append(Detection(
                    bbox=[round(v, 1) for v in xyxy],
                    confidence=round(conf, 4),
                    class_id=cls_id,
                    class_name=cls_name,
                ))

        # 按置信度降序
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def set_confidence(self, threshold: float) -> None:
        """动态调整置信度阈值"""
        self.confidence = max(0.0, min(1.0, threshold))

    def warmup(self, dummy_shape: tuple = (640, 640, 3)) -> None:
        """
        GPU 预热推理，减少首次检测延迟。
        在启动检测前调用。
        """
        dummy = np.zeros(dummy_shape, dtype=np.uint8)
        _ = self.detect(dummy)
        print("[YOLO] 预热完成")

    @property
    def model_info(self) -> dict:
        """返回模型元信息"""
        return {
            "model_path": self.model_path,
            "device": self.device,
            "input_size": self.input_size,
            "confidence": self.confidence,
            "nms_threshold": self.nms_threshold,
            "target_classes": self.target_classes,
            "total_classes": len(self.COCO_CLASSES),
        }

    @staticmethod
    def list_available_models() -> list[str]:
        """
        列出可用的预训练模型（ultralytics 会自动下载）。

        Returns:
            ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
             "yolov11n.pt", ...]
        """
        return [
            "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
            "yolov11n.pt", "yolov11s.pt", "yolov11m.pt", "yolov11l.pt", "yolov11x.pt",
        ]
