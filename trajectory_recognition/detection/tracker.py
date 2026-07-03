"""
多目标跟踪器 — Multi-Object Tracker

基于 ultralytics 内置跟踪（BoT-SORT / ByteTrack），
在 YOLO 逐帧检测结果之上分配并维护稳定的 track_id。
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Track:
    """单个目标的跟踪记录"""
    track_id: int                            # 唯一跟踪 ID
    class_name: str                          # 目标类别
    positions: list = field(default_factory=list)     # [[x, y, z], ...]  3D 坐标
    positions_2d: list = field(default_factory=list)  # [[x, y], ...]    2D 像素坐标
    timestamps: list = field(default_factory=list)     # [t, ...]
    bboxes: list = field(default_factory=list)         # [[x1,y1,x2,y2], ...]
    confidences: list = field(default_factory=list)    # 每帧置信度
    first_seen: float = 0.0                  # 首次出现时间
    last_seen: float = 0.0                   # 最后一次出现时间
    is_active: bool = True                   # 当前是否活跃
    frames_since_update: int = 0             # 距上次更新的帧数

    @property
    def confidence_avg(self) -> float:
        if not self.confidences:
            return 0.0
        return round(sum(self.confidences) / len(self.confidences), 4)

    def add_point_2d(self, x: float, y: float, bbox: list, conf: float, ts: float):
        self.positions_2d.append([x, y])
        self.bboxes.append(bbox)
        self.confidences.append(conf)
        self.timestamps.append(ts)
        self.last_seen = ts
        if not self.first_seen:
            self.first_seen = ts
        self.frames_since_update = 0

    def add_point_3d(self, x: float, y: float, z: float):
        self.positions.append([x, y, z])

    def mark_missed(self):
        self.frames_since_update += 1

    def to_summary(self) -> dict:
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "point_count": len(self.positions),
            "confidence_avg": self.confidence_avg,
            "is_active": self.is_active,
            "latest_bbox": self.bboxes[-1] if self.bboxes else [],
            "latest_position": self.positions[-1] if self.positions else [],
            "duration": round(self.last_seen - self.first_seen, 2) if self.first_seen else 0,
        }


class MultiTracker:
    """
    多目标跟踪器。

    使用 ultralytics 内置的跟踪能力：
      - BoT-SORT (默认): 高精度，适合无人机场景
      - ByteTrack: 速度快

    也支持独立使用（不依赖 ultralytics 跟踪），
    通过简单的 IOU 匹配作为降级方案。

    使用示例:
        tracker = MultiTracker(tracker_type="bytetrack")
        # 方式 1: 配合 ultralytics 内置跟踪
        results = model.track(frame, persist=True, tracker="bytetrack.yaml")
        tracker.update_from_ultralytics(results, frame_id, timestamp)

        # 方式 2: 独立跟踪
        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame_id, timestamp)
    """

    # ultralytics 支持的跟踪器配置
    TRACKER_CONFIGS = {
        "bytetrack": "bytetrack.yaml",
        "botsort": "botsort.yaml",
    }

    def __init__(
        self,
        tracker_type: str = "bytetrack",
        max_missed_frames: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
    ):
        """
        初始化跟踪器。

        Args:
            tracker_type: 跟踪算法 ("bytetrack" / "botsort" / "simple")
            max_missed_frames: 多少帧未更新后标记为丢失
            min_hits: 最少命中次数才视为有效目标
            iou_threshold: IOU 匹配阈值（simple 模式）
        """
        self.tracker_type = tracker_type
        self.max_missed_frames = max_missed_frames
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self._tracks: dict[int, Track] = {}
        self._next_id = 0
        self._frame_count = 0

    def update(
        self,
        detections: list,
        frame_id: int,
        timestamp: Optional[float] = None,
    ) -> list[Track]:
        """
        输入当前帧检测结果，更新跟踪状态。

        使用简单 IOU 匹配（不依赖 ultralytics 跟踪模块）。

        Args:
            detections: Detection 列表
            frame_id: 帧序号
            timestamp: 帧时间戳

        Returns:
            当前帧活跃的 Track 列表
        """
        self._frame_count += 1
        ts = timestamp or (frame_id / 30.0)  # 假设 30fps

        # 标记所有现有 track 为"本帧未更新"
        for t in self._tracks.values():
            t.mark_missed()

        # 匹配检测 → 已有 track
        matched_track_ids = set()
        matched_det_indices = set()

        for det_idx, det in enumerate(detections):
            best_track_id = None
            best_iou = 0.0

            # 计算 bbox 中心
            det_cx = (det.bbox[0] + det.bbox[2]) / 2
            det_cy = (det.bbox[1] + det.bbox[3]) / 2

            for tid, track in self._tracks.items():
                if tid in matched_track_ids:
                    continue
                if not track.bboxes:
                    continue

                last_bbox = track.bboxes[-1]
                iou = self._compute_iou(det.bbox, last_bbox)

                if iou > best_iou:
                    best_iou = iou
                    best_track_id = tid

            if best_iou > self.iou_threshold and best_track_id is not None:
                matched_track_ids.add(best_track_id)
                matched_det_indices.add(det_idx)
                track = self._tracks[best_track_id]
                track.add_point_2d(det_cx, det_cy, det.bbox, det.confidence, ts)
                track.class_name = det.class_name
                track.is_active = True

        # 未匹配的检测 → 新 track
        for det_idx, det in enumerate(detections):
            if det_idx in matched_det_indices:
                continue
            tid = self._next_id
            self._next_id += 1
            cx = (det.bbox[0] + det.bbox[2]) / 2
            cy = (det.bbox[1] + det.bbox[3]) / 2
            track = Track(track_id=tid, class_name=det.class_name)
            track.add_point_2d(cx, cy, det.bbox, det.confidence, ts)
            self._tracks[tid] = track

        # 清理长时间未更新的 track
        for tid in list(self._tracks.keys()):
            track = self._tracks[tid]
            if track.frames_since_update > self.max_missed_frames:
                track.is_active = False

        return self.get_active_tracks()

    def update_from_ultralytics(
        self,
        results,
        frame_id: int,
        timestamp: Optional[float] = None,
    ):
        """
        从 ultralytics track() 结果中提取跟踪数据。

        使用方式:
            results = model.track(frame, persist=True, tracker="bytetrack.yaml")
            tracks = tracker.update_from_ultralytics(results, frame_id, timestamp)
        """
        self._frame_count += 1
        ts = timestamp or (frame_id / 30.0)

        if results is None or results[0].boxes is None:
            for t in self._tracks.values():
                t.mark_missed()
            return self.get_active_tracks()

        boxes = results[0].boxes
        if boxes.id is None:
            return self.get_active_tracks()

        track_ids = boxes.id.int().tolist()
        cls_list = boxes.cls.int().tolist() if boxes.cls is not None else [0] * len(track_ids)
        conf_list = boxes.conf.tolist() if boxes.conf is not None else [1.0] * len(track_ids)
        xyxy_list = boxes.xyxy.tolist()

        for tid, cls_id, conf, xyxy in zip(track_ids, cls_list, conf_list, xyxy_list):
            if tid not in self._tracks:
                self._tracks[tid] = Track(track_id=tid, class_name=f"class_{cls_id}")
                if tid >= self._next_id:
                    self._next_id = tid + 1

            track = self._tracks[tid]
            cx = (xyxy[0] + xyxy[2]) / 2
            cy = (xyxy[1] + xyxy[3]) / 2
            track.add_point_2d(cx, cy, xyxy, conf, ts)
            track.class_name = f"class_{cls_id}"
            track.is_active = True

        # 标记未出现的 track
        active_ids = set(track_ids)
        for tid, track in self._tracks.items():
            if tid not in active_ids:
                track.mark_missed()
                if track.frames_since_update > self.max_missed_frames:
                    track.is_active = False

        return self.get_active_tracks()

    def add_3d_point(self, track_id: int, x: float, y: float, z: float):
        """为指定 track 添加三角测量后的 3D 坐标"""
        if track_id in self._tracks:
            self._tracks[track_id].add_point_3d(x, y, z)

    def get_track(self, track_id: int) -> Optional[Track]:
        return self._tracks.get(track_id)

    def get_active_tracks(self) -> list[Track]:
        return [t for t in self._tracks.values()
                if t.is_active and len(t.bboxes) >= self.min_hits]

    def get_all_tracks(self) -> list[Track]:
        return list(self._tracks.values())

    def reset(self):
        self._tracks.clear()
        self._next_id = 0
        self._frame_count = 0

    @staticmethod
    def _compute_iou(box_a: list, box_b: list) -> float:
        """计算两个 bbox 的 IOU"""
        xa = max(box_a[0], box_b[0])
        ya = max(box_a[1], box_b[1])
        xb = min(box_a[2], box_b[2])
        yb = min(box_a[3], box_b[3])

        inter_w = max(0, xb - xa)
        inter_h = max(0, yb - ya)
        inter_area = inter_w * inter_h
        if inter_area == 0:
            return 0.0

        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter_area
        return inter_area / union if union > 0 else 0.0
