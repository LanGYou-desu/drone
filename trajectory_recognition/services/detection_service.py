"""
检测服务 — Detection Service

管理双目检测会话的完整生命周期。
每个会话在后台线程中运行检测流水线。
"""

import json
import math
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

# 项目根目录
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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
        config_path = os.path.join(_PROJECT_ROOT, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[WARN] 配置文件不存在: {config_path}")
        return {}
    except Exception as e:
        print(f"[ERR] 加载配置失败: {e}")
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
    platforms = cfg.get("platforms", {})

    session = DetectionSession(
        source_a=source_a,
        source_b=source_b,
        platform_id=platform_id,
        config_snapshot={
            **(config or {}),
            "detection": det_cfg,
            "platforms": platforms,
        },
        stereo_params={},
    )
    _sessions[session.session_id] = session
    return session


def start_detection(session_id: str) -> DetectionSession:
    """在后台线程启动双目检测流水线"""
    session = _sessions.get(session_id)
    if session is None:
        raise ValueError(f"会话不存在: {session_id}")

    # 防止重复启动
    if session.status == SessionStatus.RUNNING:
        raise RuntimeError(f"会话已在运行中: {session_id}")
    if session._thread is not None and session._thread.is_alive():
        raise RuntimeError(f"会话的后台线程仍在运行: {session_id}")

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


# BBOX 颜色（按类别）
_CLASS_COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
]

def _draw_bboxes(image, detections):
    """在图像上绘制检测框和类别标签"""
    if cv2 is None:
        return
    for det in detections:
        x1, y1, x2, y2 = map(int, det.bbox)
        c = _CLASS_COLORS[det.class_id % len(_CLASS_COLORS)]
        cv2.rectangle(image, (x1, y1), (x2, y2), c, 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - th - 4), (x1 + tw + 4, y1), c, -1)
        cv2.putText(image, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


# IoU 计算统一使用 tracker 中的实现
from trajectory_recognition.detection.tracker import MultiTracker
_compute_iou = MultiTracker._compute_iou


def _to_world(pt3d, platform_pos):
    """将相机局部坐标转换为世界坐标

    输入: 相机坐标系 (X=右 Y=↓ Z=前, OpenCV标准)
    输出: 世界坐标系 (X=右 Y=↑ Z=前)
    步骤: Y轴翻转 → 平台旋转(extrinsic Yaw→Pitch→Roll) → 平台平移
    """
    if not platform_pos:
        return pt3d
    x, y_down, z = pt3d
    pitch = math.radians(platform_pos.get("pitch", 0))
    yaw   = math.radians(platform_pos.get("yaw", 0))
    roll  = math.radians(platform_pos.get("roll", 0))

    # 1. 摄像头坐标系 → 世界坐标系: Y轴翻转 (↓ → ↑)
    y = -y_down

    # 2. 旋转: 平台朝向 (extrinsic Yaw→Pitch→Roll)
    # Yaw: 绕世界Y(↑)轴, 正=右转
    x2 = x * math.cos(yaw) + z * math.sin(yaw)
    z2 = z * math.cos(yaw) - x * math.sin(yaw)
    x, z = x2, z2

    # Pitch: 绕世界X(右)轴, 正=抬头 → 前方(Z)贡献高度, 高度(Y)减少前方
    y2 = y * math.cos(pitch) + z * math.sin(pitch)
    z2 = z * math.cos(pitch) - y * math.sin(pitch)
    y, z = y2, z2

    # Roll: 绕世界Z(前)轴, 正=右滚
    # R_z(+θ) 使 Y 轴向 -X 旋转（左滚），正=右滚需用 R_z(-θ)
    x2 =  x * math.cos(roll) + y * math.sin(roll)
    y2 = -x * math.sin(roll) + y * math.cos(roll)
    x, y = x2, y2

    # 3. 平移
    return [
        x + platform_pos.get("pos_x", 0),
        y + platform_pos.get("pos_y", 0),
        z + platform_pos.get("pos_z", 0),
    ]


def _run_detection_pipeline(session: DetectionSession):
    """
    双目检测流水线（在后台线程中运行）:

    VideoProcessor_A ─→ YOLODetector ─→ (左目 dets) ─┐
                                                       ├→ match_detections → StereoTriangulator → _to_world → 3D tracks
    VideoProcessor_B ─→ YOLODetector ─→ (右目 dets) ─┘
    """
    try:
        from trajectory_recognition.detection.preprocess import VideoProcessor
        from trajectory_recognition.detection.engine import YOLODetector
        from trajectory_recognition.detection.tracker import MultiTracker
        from trajectory_recognition.detection.stereo import StereoTriangulator, StereoParams

        det_cfg = session.config_snapshot.get("detection", {})
        platforms = session.config_snapshot.get("platforms", {})
        pid = session.platform_id
        plat_cfg = platforms.get(pid, platforms.get("visible", {})) if platforms else {}

        # 提取立体参数
        stereo_keys = ["focal_length_px", "baseline", "fov_horizontal",
                       "fov_vertical", "resolution_width", "resolution_height"]
        stereo_cfg = {k: plat_cfg[k] for k in stereo_keys if k in plat_cfg}
        session.stereo_params = stereo_cfg

        # 提取位置朝向
        platform_pos = {k: plat_cfg.get(k, 0) for k in ["pos_x", "pos_y", "pos_z", "pitch", "yaw", "roll"]}

        # 目标类别 ID（用于双目匹配时的类别约束）
        # 优先取显式配置，否则取 target_classes 的第一个值
        target_class_id = det_cfg.get("target_class_id", None)
        if target_class_id is None:
            tc = det_cfg.get("target_classes", None)
            if tc and len(tc) > 0:
                target_class_id = tc[0]

        # 初始化组件
        detector = YOLODetector(
            model_path=det_cfg.get("model", "models/yolo/yolov8n.pt"),
            confidence=det_cfg.get("confidence_threshold", 0.5),
            nms_threshold=det_cfg.get("nms_threshold", 0.45),
            device=det_cfg.get("device", "cpu"),
            input_size=(
                det_cfg.get("input_width", 640),
                det_cfg.get("input_height", 640),
            ),
            target_classes=det_cfg.get("target_classes", None),
        )
        tracker = MultiTracker(
            tracker_type=det_cfg.get("tracker", "bytetrack"),
        )
        session._tracker = tracker
        stereo = StereoTriangulator(StereoParams(**stereo_cfg)) if stereo_cfg else None

        frame_interval = det_cfg.get("frame_interval", 5)
        proc_a = VideoProcessor(frame_interval=frame_interval)
        proc_b = VideoProcessor(frame_interval=frame_interval)

        # 获取视频信息 + 时间同步校验
        info_a = proc_a.get_video_info(session.source_a)
        info_b = proc_b.get_video_info(session.source_b)
        session.total_frames = min(info_a["total_frames"], info_b["total_frames"])

        # 检查帧率是否一致（时间同步的基本前提）
        fps_a = info_a.get("fps", 0)
        fps_b = info_b.get("fps", 0)
        if abs(fps_a - fps_b) > 0.5:  # 允许 0.5fps 容差
            print(f"[警告] 双目视频帧率不一致: A={fps_a}fps, B={fps_b}fps — 可能导致时间同步偏差")

        # 预热
        detector.warmup()

        # 双路同步提取帧
        gen_a = proc_a.extract_frames(session.source_a)
        gen_b = proc_b.extract_frames(session.source_b)

        # 3D-track 关联的 IoU 阈值
        ASSOCIATION_IOU_THRESHOLD = 0.3

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

            # 时间同步校验：两帧时间戳相差过大时发出警告
            time_diff = abs(frame_a.timestamp - frame_b.timestamp)
            if time_diff > 0.1:  # 超过 100ms 视为不同步
                print(f"[警告] 帧时间戳偏差较大: Δt={time_diff:.2f}s "
                      f"(frame_a={frame_a.timestamp:.3f}, frame_b={frame_b.timestamp:.3f})")

            # YOLO 检测（左右目分别推理）
            dets_a = detector.detect(frame_a.image)
            dets_b = detector.detect(frame_b.image)

            # 在预览帧上绘制检测框 + 类别标签
            _draw_bboxes(frame_a.image, dets_a)
            _draw_bboxes(frame_b.image, dets_b)

            # 缓存预览帧（JPEG 编码）
            if cv2 is None:
                raise ImportError("opencv-python 未安装")
            _, buf_a = cv2.imencode('.jpg', frame_a.image, [cv2.IMWRITE_JPEG_QUALITY, 60])
            _, buf_b = cv2.imencode('.jpg', frame_b.image, [cv2.IMWRITE_JPEG_QUALITY, 60])
            session._frame_a = buf_a.tobytes()
            session._frame_b = buf_b.tobytes()

            # 跟踪（基于左目）
            tracks = tracker.update(dets_a, frame_a.frame_id, frame_a.timestamp)

            # 双目匹配 + 三角测量 + 世界坐标变换
            if stereo and dets_a and dets_b:
                matches = stereo.match_detections(dets_a, dets_b, target_class_id=target_class_id)
                # 当前帧时间戳（用于 3D 点时间戳对齐）
                frame_ts = frame_a.timestamp

                for dl, dr, pt3d in matches:
                    if pt3d is None:
                        continue
                    # 世界坐标变换
                    pt3d = _to_world(pt3d, platform_pos)

                    # 3D 点与 track 关联 — 使用 IoU 匹配代替固定像素阈值
                    best_track = None
                    best_iou = 0.0
                    for track in tracks:
                        if not track.bboxes:
                            continue
                        last_bbox = track.bboxes[-1]
                        iou = _compute_iou(dl.bbox, last_bbox)
                        if iou > best_iou:
                            best_iou = iou
                            best_track = track

                    if best_track is not None and best_iou >= ASSOCIATION_IOU_THRESHOLD:
                        best_track.add_point_3d(*pt3d, ts=frame_ts)
                        session.points_3d += 1

            session.track_count = len(tracker.get_active_tracks())
            # 进度以帧数较少的一方为准
            session.progress = min(
                session.current_frame_a / max(session.total_frames, 1),
                session.current_frame_b / max(session.total_frames, 1),
            )

        session.status = SessionStatus.COMPLETED

    except Exception as e:
        session.status = SessionStatus.ERROR
        session.error_message = str(e)
        print(f"[ERR] 检测流水线异常: {e}")
    finally:
        # 确保视频资源释放
        for gen in (gen_a, gen_b):
            try:
                gen.close()
            except Exception:
                pass
        session.finished_at = time.time()
        session._thread = None


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
