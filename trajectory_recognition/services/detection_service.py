"""
检测服务 — Detection Service

管理双目检测会话的完整生命周期。
每个会话在后台线程中运行检测流水线。
"""

import json
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from uuid import uuid4

try:
    import cv2
except ImportError:
    cv2 = None


class SessionStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class DetectionSession:
    session_id: str = field(default_factory=lambda: uuid4().hex[:8])
    status: SessionStatus = SessionStatus.IDLE
    progress: float = 0.0
    current_frame_a: int = 0
    current_frame_b: int = 0
    total_frames: int = 0
    track_count: int = 0
    points_3d: int = 0
    platform_id: str = "visible"
    source_a: str = ""
    source_b: str = ""
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error_message: str = ""
    config_snapshot: dict = field(default_factory=dict)
    stereo_params: dict = field(default_factory=dict)

    # 非序列化：运行时对象
    _detector: Optional[object] = field(default=None, repr=False)
    _tracker: Optional[object] = field(default=None, repr=False)
    _stereo: Optional[object] = field(default=None, repr=False)
    _thread: Optional[threading.Thread] = field(default=None, repr=False)
    _pause_event: Optional[threading.Event] = field(default=None, repr=False)
    _frame_a: Optional[bytes] = field(default=None, repr=False)  # 左目最新帧 JPEG
    _frame_b: Optional[bytes] = field(default=None, repr=False)  # 右目最新帧 JPEG
    _stop_event: Optional[threading.Event] = field(default=None, repr=False)

    @property
    def duration(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.finished_at or time.time()
        return end - self.started_at

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "status": self.status.value,
            "progress": round(self.progress, 3),
            "current_frame_a": self.current_frame_a,
            "current_frame_b": self.current_frame_b,
            "total_frames": self.total_frames,
            "track_count": self.track_count,
            "points_3d": self.points_3d,
            "platform_id": self.platform_id,
            "source_a": self.source_a,
            "source_b": self.source_b,
            "duration": round(self.duration, 1),
            "error": self.error_message,
        }


# ── 全局会话注册表 ──────────────────────────────────

_sessions: dict[str, DetectionSession] = {}
_active_session_id: Optional[str] = None


def _load_config():
    """加载全局配置"""
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def create_session(
    source_a: str,
    source_b: str,
    platform_id: str = "visible",
    config: Optional[dict] = None,
) -> DetectionSession:
    """创建双目检测会话"""
    cfg = _load_config()
    det_cfg = cfg.get("detection", {})
    stereo_cfg = cfg.get("stereo", {})

    session = DetectionSession(
        source_a=source_a,
        source_b=source_b,
        platform_id=platform_id,
        config_snapshot={
            **(config or {}),
            "detection": det_cfg,
        },
        stereo_params=stereo_cfg,
    )
    _sessions[session.session_id] = session
    return session


def start_detection(session_id: str) -> DetectionSession:
    """在后台线程启动双目检测流水线"""
    session = _sessions.get(session_id)
    if session is None:
        raise ValueError(f"会话不存在: {session_id}")

    global _active_session_id
    _active_session_id = session_id

    session.status = SessionStatus.RUNNING
    session.started_at = time.time()
    session._pause_event = threading.Event()
    session._pause_event.set()  # 初始非暂停
    session._stop_event = threading.Event()

    session._thread = threading.Thread(
        target=_run_detection_pipeline,
        args=(session,),
        daemon=True,
    )
    session._thread.start()
    return session


def _run_detection_pipeline(session: DetectionSession):
    """
    双目检测流水线（在后台线程中运行）:

    VideoProcessor_A ─→ YOLODetector ─→ (左目 dets) ─┐
                                                        ├→ match_detections → StereoTriangulator → 3D tracks
    VideoProcessor_B ─→ YOLODetector ─→ (右目 dets) ─┘
    """
    try:
        from trajectory_recognition.detection.preprocess import VideoProcessor
        from trajectory_recognition.detection.engine import YOLODetector
        from trajectory_recognition.detection.tracker import MultiTracker
        from trajectory_recognition.detection.stereo import StereoTriangulator, StereoParams

        det_cfg = session.config_snapshot.get("detection", {})
        stereo_cfg = session.stereo_params

        # 初始化组件
        detector = YOLODetector(
            model_path=det_cfg.get("model", "models/yolov8n.pt"),
            confidence=det_cfg.get("confidence_threshold", 0.5),
            nms_threshold=det_cfg.get("nms_threshold", 0.45),
            device=det_cfg.get("device", "cpu"),
            input_size=(
                det_cfg.get("input_width", 640),
                det_cfg.get("input_height", 640),
            ),
        )
        tracker = MultiTracker(
            tracker_type=det_cfg.get("tracker", "bytetrack"),
        )
        session._tracker = tracker
        stereo = StereoTriangulator(StereoParams(**stereo_cfg)) if stereo_cfg else None

        frame_interval = det_cfg.get("frame_interval", 5)
        proc_a = VideoProcessor(frame_interval=frame_interval)
        proc_b = VideoProcessor(frame_interval=frame_interval)

        # 获取总帧数
        info_a = proc_a.get_video_info(session.source_a)
        info_b = proc_b.get_video_info(session.source_b)
        session.total_frames = max(info_a["total_frames"], info_b["total_frames"])

        # 预热
        detector.warmup()

        # 双路同步提取帧
        gen_a = proc_a.extract_frames(session.source_a)
        gen_b = proc_b.extract_frames(session.source_b)

        while True:
            # 检查停止/暂停
            if session._stop_event.is_set():
                break
            session._pause_event.wait()

            try:
                frame_a = next(gen_a)
                frame_b = next(gen_b)
            except StopIteration:
                break

            session.current_frame_a = frame_a.frame_id
            session.current_frame_b = frame_b.frame_id

            # 缓存预览帧（JPEG 编码）
            if cv2 is None:
                raise ImportError("opencv-python 未安装")
            _, buf_a = cv2.imencode('.jpg', frame_a.image, [cv2.IMWRITE_JPEG_QUALITY, 60])
            _, buf_b = cv2.imencode('.jpg', frame_b.image, [cv2.IMWRITE_JPEG_QUALITY, 60])
            session._frame_a = buf_a.tobytes()
            session._frame_b = buf_b.tobytes()

            # YOLO 检测（左右目分别推理）
            dets_a = detector.detect(frame_a.image)
            dets_b = detector.detect(frame_b.image)

            # 跟踪（基于左目）
            tracks = tracker.update(dets_a, frame_a.frame_id, frame_a.timestamp)

            # 双目匹配 + 三角测量
            if stereo and dets_a and dets_b:
                matches = stereo.match_detections(dets_a, dets_b)
                for dl, dr, pt3d in matches:
                    if pt3d is None:
                        continue
                    # 找到对应的 track 并添加 3D 点
                    for track in tracks:
                        if not track.bboxes:
                            continue
                        last_bbox = track.bboxes[-1]
                        # 匹配：bbox 中心接近
                        cxl = (dl.bbox[0] + dl.bbox[2]) / 2
                        tcx = (last_bbox[0] + last_bbox[2]) / 2
                        if abs(cxl - tcx) < 10:
                            track.add_point_3d(*pt3d)
                            session.points_3d += 1
                            break

            session.track_count = len(tracker.get_active_tracks())
            session.progress = min(
                session.current_frame_a / max(session.total_frames, 1),
                session.current_frame_b / max(session.total_frames, 1),
            )

        session.status = SessionStatus.COMPLETED

    except Exception as e:
        session.status = SessionStatus.ERROR
        session.error_message = str(e)
    finally:
        session.finished_at = time.time()
        session._thread = None
        # 完成后保持 session 可查询，不清空 _active_session_id
        # 只有新 session 启动或手动 stop 才切换

        # 自动保存（tracks_to_dat 内部已含 auto_backup）
        if session.config_snapshot.get("detection", {}).get("auto_save", True):
            try:
                from trajectory_recognition.services.data_bridge import tracks_to_dat
                trk = session._tracker
                if trk:
                    tracks_to_dat(
                        trk.get_all_tracks(),
                        platform_id=session.platform_id,
                        auto_backup=True,
                    )
            except Exception:
                pass


def pause_detection(session_id: str):
    session = _sessions.get(session_id)
    if session and session._pause_event:
        session._pause_event.clear()
        session.status = SessionStatus.PAUSED


def resume_detection(session_id: str):
    session = _sessions.get(session_id)
    if session and session._pause_event:
        session._pause_event.set()
        session.status = SessionStatus.RUNNING


def stop_detection(session_id: str):
    session = _sessions.get(session_id)
    if session and session._stop_event:
        session._stop_event.set()
        session.status = SessionStatus.COMPLETED


def get_session(session_id: str) -> Optional[DetectionSession]:
    return _sessions.get(session_id)


def get_active_session() -> Optional[DetectionSession]:
    global _active_session_id
    if _active_session_id:
        return _sessions.get(_active_session_id)
    return None


def list_sessions(status: Optional[SessionStatus] = None) -> list[DetectionSession]:
    sessions = list(_sessions.values())
    if status:
        sessions = [s for s in sessions if s.status == status]
    return sessions


def delete_session(session_id: str) -> bool:
    if session_id in _sessions:
        session = _sessions[session_id]
        if session.status == SessionStatus.RUNNING:
            stop_detection(session_id)
        del _sessions[session_id]
        global _active_session_id
        if _active_session_id == session_id:
            _active_session_id = None
        return True
    return False
