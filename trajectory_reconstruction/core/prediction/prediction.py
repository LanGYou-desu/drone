"""
轨迹预测算法

提供两种预测方法:
  1. generate_prediction (线性外推)  — 简单快速，始终可用
  2. generate_prediction_hybrid     — Phy-ODE-Diffusion 模型，高精度物理约束预测

预测入口函数根据模型是否加载自动选择方法。
"""

import os
from typing import Optional

import torch

from trajectory_reconstruction.core.config.config_manager import _PROJECT_ROOT

# ── 模型缓存 ──────────────────────────────────────────

_model_cache: Optional["PhyODEDiffusion"] = None
_model_device: str = "cpu"


def _get_model_path() -> str:
    """返回最新/最佳模型权重路径，优先查找带描述的最佳检查点"""
    model_dir = os.path.join(_PROJECT_ROOT, "models", "hybrid_predictor")
    if not os.path.isdir(model_dir):
        return os.path.join(model_dir, "phy_ode_diffusion.pt")

    # 优先查找最佳模型
    best_files = sorted(
        [f for f in os.listdir(model_dir) if f.startswith("phy_ode_diffusion_best")],
        reverse=True,
    )
    if best_files:
        return os.path.join(model_dir, best_files[0])

    # 次选最新带描述的检查点
    all_files = sorted(
        [f for f in os.listdir(model_dir) if f.startswith("phy_ode_diffusion_s")],
        reverse=True,
    )
    if all_files:
        return os.path.join(model_dir, all_files[0])

    # 回退到旧命名
    return os.path.join(model_dir, "phy_ode_diffusion.pt")


def _get_hybrid_config() -> dict:
    """从 config.json 读取混合模型配置，缺失字段用默认值填充"""
    try:
        from trajectory_reconstruction.core.config.config_manager import ensure_config
        cfg = ensure_config()
        return cfg.get("hybrid_model", {})
    except Exception:
        return {}


def load_hybrid_model(device: str = None) -> Optional["PhyODEDiffusion"]:
    """
    加载 Phy-ODE-Diffusion 模型（含缓存）。

    从 config.json → hybrid_model 读取配置参数。
    若权重文件不存在，返回 None。

    Args:
        device: 推理设备，None 则从 config.json 读取
    """
    global _model_cache, _model_device

    model_path = _get_model_path()
    if not os.path.isfile(model_path):
        print(f"[预测] 混合模型权重不存在: {model_path}，回退到线性外推")
        return None

    # 从 config.json 读取配置
    h_cfg = _get_hybrid_config()
    if device is None:
        device = h_cfg.get("device", "cpu")
    if not h_cfg.get("enabled", True):
        print("[预测] 混合模型已禁用 (hybrid_model.enabled=false)，回退到线性外推")
        return None

    # 设备回退：CUDA 不可用时自动使用 CPU
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[预测] CUDA 不可用，回退到 CPU")
        device = "cpu"

    if _model_cache is not None and _model_device == device:
        return _model_cache

    try:
        from trajectory_reconstruction.core.prediction.hybrid import PhyODEDiffusion

        device_obj = torch.device(device)
        ckpt = torch.load(model_path, map_location=device_obj, weights_only=False)

        # 用 config.json 的参数创建模型（物理约束、引导强度等）
        model = PhyODEDiffusion(
            v_max=h_cfg.get("v_max", 30.0),
            a_max=h_cfg.get("a_max", 30.0),
            z_min=h_cfg.get("z_min", 0.0),
            guidance_eta=h_cfg.get("guidance_eta", 0.1),
            n_inference_steps=h_cfg.get("inference_steps", 50),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device_obj)
        model.eval()
        model.diffusion.scheduler.to(device_obj)

        _model_cache = model
        _model_device = device
        print(f"[预测] 已加载混合模型: {model_path} "
              f"({model.get_model_info()['total_params']:,} 参数, "
              f"device={device}, v_max={model.v_max})")
        return model

    except ImportError:
        print("[预测] hybrid 模块不可用，回退到线性外推")
        return None
    except Exception as e:
        print(f"[预测] 模型加载失败: {e}，回退到线性外推")
        return None


# ── 线性外推（保留）──────────────────────────────────

def generate_prediction(
    points: list[list[float]],
    timestamps: list[float],
    num_points: int = 5,
    time_step: float = 0.5,
) -> tuple[list[list[float]], list[float]]:
    """
    基于最后两点线性外推，生成未来轨迹预测点（兜底方案）。

    Args:
        points:     历史轨迹点 [[x, y, z], ...]，至少 2 个点
        timestamps: 各点时间戳 [t, ...]，长度需与 points 一致
        num_points: 预测点数
        time_step:  预测点之间的时间间隔（秒）

    Returns:
        pred_points: 预测坐标 [[x, y, z], ...]
        pred_times:  对应时间戳 [t, ...]
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


# ── 混合模型预测 ─────────────────────────────────────

def generate_prediction_hybrid(
    points: list[list[float]],
    timestamps: list[float],
    num_points: int = 5,
    time_step: float = 0.5,
    device: str = "cpu",
) -> tuple[list[list[float]], list[float]]:
    """
    使用 Phy-ODE-Diffusion 模型进行轨迹预测。

    自动加载已训练的模型权重（从 models/hybrid_predictor/phy_ode_diffusion.pt）。
    若模型不可用，自动回退到线性外推。

    Args:
        points:     历史轨迹点 [[x, y, z], ...]，至少 5 个点（模型需要足够上下文）
        timestamps: 各点时间戳 [t, ...]
        num_points: 预测点数
        time_step:  预测时间步长（秒）
        device:     推理设备 ("cpu" / "cuda:0")

    Returns:
        pred_points: 预测坐标 [[x, y, z], ...]
        pred_times:  对应时间戳 [t, ...]
    """
    model = load_hybrid_model(device)

    if model is None:
        # 回退到线性外推
        return generate_prediction(points, timestamps, num_points, time_step)

    # 至少需要 10 个点才能得到有意义的 Transformer 编码
    if len(points) < 10:
        print("[预测] 历史点不足 (< 10)，回退到线性外推")
        return generate_prediction(points, timestamps, num_points, time_step)

    try:
        pred_positions, pred_times = model.predict(
            timestamps=timestamps,
            positions=points,
            num_points=num_points,
            time_step=time_step,
            device=device,
        )
        return pred_positions, pred_times
    except Exception as e:
        print(f"[预测] 混合模型推理失败: {e}，回退到线性外推")
        return generate_prediction(points, timestamps, num_points, time_step)


def is_hybrid_model_available() -> bool:
    """检查混合模型权重文件是否存在"""
    return os.path.isfile(_get_model_path())
