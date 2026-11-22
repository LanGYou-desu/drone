"""
轨迹预测入口

提供两种预测方法:
  1. generate_linear_prediction  — 匀速线性外推，简单快速，始终可用（兜底）
  2. generate_kinflow_prediction — KinFlow 模型，物理约束并行非自回归预测

KinFlow（学习式运动学先验 + 并行非自回归细化）:
  连续时间编码器 → 运动学走廊先验 → 走廊初始化流匹配 → 可微物理投影器。
权重缺失、模型被禁用或推理异常时，自动回退到线性外推。
"""

import os
from typing import Optional

import torch

from trajectory_reconstruction.core.config.config_manager import _PROJECT_ROOT

# ── 模型缓存 ──────────────────────────────────────────

_model_cache: Optional["KinFlow"] = None
_model_device: str = "cpu"
_model_path: str = ""

# 可由 config.json → kinflow_model 覆盖的构造参数
_OVERRIDABLE_KEYS = (
    "v_h_max", "v_up_max", "v_down_max", "y_min", "y_max",
    "max_tilt", "g", "n_euler_steps", "sigma", "project",
)


def _find_kinflow_weights() -> str:
    """
    查找可用的 KinFlow 权重，优先级:
      1. train/kinflow_predictor/train_result/best/kinflow_best_s2|s1.pt
      2. models/kinflow/kinflow_best_s*.pt
      3. models/kinflow/kinflow_s*_v{loss}.pt（按 loss 升序取最优）
      4. models/kinflow/kinflow.pt（兼容手写文件名）
    """
    best_central = os.path.join(_PROJECT_ROOT, "train", "kinflow_predictor", "train_result", "best")
    if os.path.isdir(best_central):
        for stage in (2, 1):
            candidate = os.path.join(best_central, f"kinflow_best_s{stage}.pt")
            if os.path.isfile(candidate):
                return candidate

    model_dir = os.path.join(_PROJECT_ROOT, "models", "kinflow")
    if os.path.isdir(model_dir):
        for stage in (2, 1):
            candidate = os.path.join(model_dir, f"kinflow_best_s{stage}.pt")
            if os.path.isfile(candidate):
                return candidate

        checkpoints = [f for f in os.listdir(model_dir)
                       if f.startswith("kinflow_s") and f.endswith(".pt")]
        if checkpoints:
            def _loss_of(fname: str) -> float:
                try:
                    return float(fname.rsplit("_v", 1)[-1].rstrip(".pt"))
                except (ValueError, IndexError):
                    return float("inf")
            checkpoints.sort(key=_loss_of)
            return os.path.join(model_dir, checkpoints[0])

        plain = os.path.join(model_dir, "kinflow.pt")
        if os.path.isfile(plain):
            return plain

    return os.path.join(model_dir, "kinflow.pt")


def _get_kinflow_config() -> dict:
    """读取 config.json → kinflow_model（失败时返回空字典）。"""
    try:
        from trajectory_reconstruction.core.config.config_manager import ensure_config
        return ensure_config().get("kinflow_model", {})
    except Exception:
        return {}


def load_kinflow_model(device: str = None) -> Optional["KinFlow"]:
    """
    加载 KinFlow 模型（含进程内缓存）。

    架构参数从检查点的 model_info 读取，物理边界/采样步数/设备可被
    config.json → kinflow_model 覆盖。权重不存在或禁用时返回 None。

    Args:
        device: 推理设备，None 则读 config.json
    """
    global _model_cache, _model_device, _model_path

    path = _find_kinflow_weights()
    if not os.path.isfile(path):
        print(f"[预测] KinFlow 权重不存在: {path}，回退到线性外推", flush=True)
        return None

    cfg = _get_kinflow_config()
    if not cfg.get("enabled", True):
        print("[预测] KinFlow 已禁用 (kinflow_model.enabled=false)，回退到线性外推", flush=True)
        return None

    if device is None:
        device = cfg.get("device", "cpu")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        print("[预测] CUDA 不可用，KinFlow 回退到 CPU")
        device = "cpu"

    if _model_cache is not None and _model_device == device and _model_path == path:
        return _model_cache

    try:
        from trajectory_reconstruction.core.prediction.kinflow import KinFlow

        device_obj = torch.device(device)
        ckpt = torch.load(path, map_location=device_obj, weights_only=False)
        info = ckpt.get("model_info", {}) or {}

        # 架构参数: 检查点优先，缺失用模型默认值
        # 架构参数必须与训练时完全一致：尤其 ctx_len / sample_dt 决定推理时历史
        # 重采样窗口，漏传会让模型用更短的历史（尺度与训练不符）导致预测退化。
        arch = {k: info[k] for k in (
            "d_feat", "d_context", "n_head", "n_layers", "dim_feedforward", "dropout",
            "corridor_hidden", "d_z", "flow_hidden", "flow_hidden_layers",
            "tau_emb_dim", "time_emb_dim", "k_max", "n_project_iters",
            "ctx_len", "sample_dt", "max_native_steps",
        ) if k in info}

        # 物理/采样参数: config.json 优先
        runtime = {}
        for key in _OVERRIDABLE_KEYS:
            if key in cfg:
                runtime[key] = cfg[key]
        for key in _OVERRIDABLE_KEYS:
            if key not in runtime and key in info:
                runtime[key] = info[key]

        model = KinFlow(**arch, **runtime)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device_obj)
        model.eval()

        _model_cache = model
        _model_device = device
        _model_path = path
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[预测] 已加载 KinFlow: {path} ({n_params:,} 参数, device={device}, "
              f"euler={model.n_euler_steps}, v_h_max={model.bounds.v_h_max})", flush=True)
        return model

    except ImportError:
        print("[预测] kinflow 模块不可用，回退到线性外推")
        return None
    except Exception as exc:
        print(f"[预测] KinFlow 加载失败: {exc}，回退到线性外推", flush=True)
        return None


# ── 线性外推（兜底）──────────────────────────────────

def generate_linear_prediction(
    points: list[list[float]],
    timestamps: list[float],
    num_points: int = 5,
    time_step: float = 0.5,
) -> tuple[list[list[float]], list[float]]:
    """
    基于最后两点的匀速外推（兜底方案）。

    Args:
        points:     历史轨迹点 [[x, y, z], ...]，至少 2 个点
        timestamps: 各点时间戳 [t, ...]，长度与 points 一致
        num_points: 预测点数
        time_step:  预测点之间的时间间隔（秒）

    Returns:
        pred_points: 预测坐标 [[x, y, z], ...]（含锚点，共 num_points+1 个）
        pred_times:  对应时间戳
    """
    if len(points) < 2:
        return [], []

    last = points[-1]
    second_last = points[-2]
    last_t = timestamps[-1] if timestamps else 0.0
    second_last_t = timestamps[-2] if len(timestamps) >= 2 else last_t - 1.0

    dt = last_t - second_last_t
    if dt <= 0:
        dt = 1.0

    vx = (last[0] - second_last[0]) / dt
    vy = (last[1] - second_last[1]) / dt
    vz = (last[2] - second_last[2]) / dt

    pred_points: list[list[float]] = [list(last)]
    pred_times: list[float] = [last_t]

    for i in range(1, num_points + 1):
        t = last_t + i * time_step
        pred_points.append([
            last[0] + vx * (t - last_t),
            last[1] + vy * (t - last_t),
            last[2] + vz * (t - last_t),
        ])
        pred_times.append(t)

    return pred_points, pred_times


# ── KinFlow 预测 ─────────────────────────────────────

def generate_kinflow_prediction_with_confidence(
    points: list[list[float]],
    timestamps: list[float],
    num_points: int = 5,
    time_step: float = 0.5,
    device: str = "cpu",
    sigma: float = None,
    n_samples: int = 1,
) -> tuple[list[list[float]], list[float], list[float]]:
    """
    与 generate_kinflow_prediction 相同，但额外返回逐点置信度。

    视界 ≤ 模型可靠视界（k_max × sample_dt，默认 2 s）内为 1.0；
    更长片段由运动模型外推，置信度随时间衰减（最低 0.15）。
    """
    model = load_kinflow_model(device)
    if model is None or len(points) < 5:
        pts, ts = generate_linear_prediction(points, timestamps, num_points, time_step)
        return pts, ts, [1.0] * len(pts)
    try:
        return model.predict_with_confidence(
            timestamps=timestamps, positions=points, num_points=num_points,
            time_step=time_step, device=device, sigma=sigma, n_samples=n_samples,
        )
    except Exception as exc:
        print(f"[预测] KinFlow 推理失败: {exc}，回退到线性外推")
        pts, ts = generate_linear_prediction(points, timestamps, num_points, time_step)
        return pts, ts, [1.0] * len(pts)


def generate_kinflow_prediction(
    points: list[list[float]],
    timestamps: list[float],
    num_points: int = 5,
    time_step: float = 0.5,
    device: str = "cpu",
    sigma: float = None,
    n_samples: int = 1,
) -> tuple[list[list[float]], list[float]]:
    """
    使用 KinFlow 做并行（非自回归）轨迹预测。

    Args:
        points:     历史轨迹点 [[x, y, z], ...]，建议 ≥ 5
        timestamps: 各点时间戳 [t, ...]
        num_points: 预测点数
        time_step:  未来相邻预测点时间间隔（秒）
        device:     推理设备 ("cpu" / "cuda:0")
        sigma:      起点噪声（0/None = 确定性均值预测，>0 = 单峰概率采样）
        n_samples:  概率样本数（取第一条返回）

    Returns:
        pred_points: 预测坐标（含锚点，共 num_points+1 个）
        pred_times:  对应时间戳
    """
    if len(points) < 5:
        print("[预测] 历史点不足 (< 5)，回退到线性外推")
        return generate_linear_prediction(points, timestamps, num_points, time_step)
    pts, ts, _ = generate_kinflow_prediction_with_confidence(
        points, timestamps, num_points=num_points, time_step=time_step,
        device=device, sigma=sigma, n_samples=n_samples,
    )
    return pts, ts


def is_kinflow_model_available() -> bool:
    """检查是否存在可用的 KinFlow 权重文件。"""
    return os.path.isfile(_find_kinflow_weights())


__all__ = [
    "load_kinflow_model",
    "generate_linear_prediction",
    "generate_kinflow_prediction",
    "generate_kinflow_prediction_with_confidence",
    "is_kinflow_model_available",
]
