"""
KinFlow 轨迹数据集 — 滑动窗口采样 + 逐轨迹标准化

支持:
  - .npz 合成轨迹（positions + timestamps）
  - .dat 真实轨迹（每行 x y z t）
  - 滑动窗口 (context, target) 采样，窗口间按 stride = min(C/2, K/2) 重叠
  - 训练期增强：绕垂直轴旋转（爬升角 < 30° 时）、0.9~1.1 倍随机缩放、小高斯噪声

标准化: 逐轨迹逐轴 Z-score。统计量 *只* 由历史窗口计算（推理时也只有历史可见），
并对每轴标准差设置物理下限 POSITION_STD_FLOOR，避免平飞轨迹的垂直/侧向
厘米级变化把归一化坐标放大几十倍、破坏流匹配的数值尺度。
"""

import os

import numpy as np
import torch
from numpy.random import default_rng
from torch.utils.data import Dataset

from trajectory_reconstruction.core.math_utils import estimate_velocity
from trajectory_reconstruction.core.prediction.kinflow.model import POSITION_STD_FLOOR


# ── 轨迹加载 ──────────────────────────────────────────

def load_dat_trajectory(file_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    从 .dat 文件加载轨迹（每行 "x y z t"，空格分隔）。

    Returns:
        positions: (N, 3), timestamps: (N,)
    """
    points, timestamps = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 4:
                try:
                    x, y, z, t = map(float, parts[:4])
                except ValueError:
                    continue
                points.append([x, y, z])
                timestamps.append(t)

    if len(points) < 3:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    return np.array(points, dtype=np.float32), np.array(timestamps, dtype=np.float32)


def load_trajectory_set(
    data_dir: str = None,
    synthetic_dir: str = None,
    max_files: int = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    加载目录下所有可用轨迹（.npz 优先，兼容 .dat）。

    Args:
        data_dir:     真实 .dat 轨迹目录（可选）
        synthetic_dir: 合成轨迹目录（.npz / .dat）
        max_files:    最多加载文件数

    Returns:
        [(positions, timestamps), ...]
    """
    trajectories = []

    for directory in (data_dir, synthetic_dir):
        if not directory or not os.path.isdir(directory):
            continue
        for fname in sorted(os.listdir(directory)):
            fpath = os.path.join(directory, fname)
            if fname.endswith('.npz'):
                data = np.load(fpath)
                pos = np.asarray(data['positions'], dtype=np.float32)
                ts = np.asarray(data['timestamps'], dtype=np.float32)
                if len(pos) >= 5 and len(ts) == len(pos):
                    trajectories.append((pos, ts))
            elif fname.endswith('.dat'):
                pos, ts = load_dat_trajectory(fpath)
                if len(pos) >= 5:
                    trajectories.append((pos, ts))

    if max_files and len(trajectories) > max_files:
        trajectories = trajectories[:max_files]

    return trajectories


# ── 滑动窗口 ──────────────────────────────────────────

def build_sliding_windows(
    positions: np.ndarray,
    timestamps: np.ndarray,
    ctx_len: int = 20,
    tgt_len: int = 10,
) -> list[dict]:
    """
    从单条轨迹中按 stride = min(C/2, K/2) 提取 (上下文, 目标) 窗口。
    """
    N = len(positions)
    min_len = ctx_len + tgt_len
    if N < min_len:
        return []

    # 数据卫生: 非有限值或时间戳非严格递增的轨迹直接跳过
    if not (np.isfinite(positions).all() and np.isfinite(timestamps).all()):
        return []
    if np.any(np.diff(timestamps) <= 0):
        return []

    stride = max(1, min(ctx_len // 2, tgt_len // 2))
    windows = []

    for start in range(0, N - min_len + 1, stride):
        end_ctx = start + ctx_len
        end_tgt = end_ctx + tgt_len

        ctx_pos = positions[start:end_ctx]
        ctx_t = timestamps[start:end_ctx]
        ctx_dt = np.zeros(ctx_len, dtype=np.float32)
        ctx_dt[1:] = ctx_t[1:] - ctx_t[:-1]

        tgt_pos = positions[end_ctx:end_tgt]
        tgt_t = timestamps[end_ctx:end_tgt]

        windows.append({
            "ctx_pos": ctx_pos,
            "ctx_t": ctx_t,
            "ctx_dt": ctx_dt,
            "ctx_vel": estimate_velocity(ctx_pos, ctx_t),
            "tgt_pos": tgt_pos,
            "tgt_t": tgt_t,
            "tgt_vel": estimate_velocity(tgt_pos, tgt_t),
        })

    return windows


def _augment(
    ctx_pos, ctx_t, ctx_dt, ctx_vel, tgt_pos, tgt_vel,
) -> tuple:
    """
    训练期增强（论文 §4.1）:
      - 绕垂直轴（Y 轴）随机旋转，仅对爬升角 < 30° 的轨迹施加
      - 0.9~1.1 倍随机缩放
      - 相对数据标准差 0.5% 的高斯噪声
    """
    all_vel = np.concatenate([ctx_vel, tgt_vel], axis=0)
    horizontal = np.linalg.norm(all_vel[:, [0, 2]], axis=1).mean()
    vertical = np.abs(all_vel[:, 1]).mean()
    climb_angle = np.arctan2(vertical, max(horizontal, 1e-6))

    if climb_angle < np.radians(30):
        angle = np.random.uniform(0, 2 * np.pi)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot = np.array([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]], dtype=np.float32)
        ctx_pos = ctx_pos @ rot.T
        ctx_vel = ctx_vel @ rot.T
        tgt_pos = tgt_pos @ rot.T
        tgt_vel = tgt_vel @ rot.T

    scale = np.random.uniform(0.9, 1.1)
    ctx_pos = ctx_pos * scale
    ctx_vel = ctx_vel * scale
    tgt_pos = tgt_pos * scale
    tgt_vel = tgt_vel * scale

    pos_std = np.std(np.concatenate([ctx_pos, tgt_pos], axis=0), axis=0).clip(min=1e-3)
    ctx_pos = ctx_pos + np.random.randn(*ctx_pos.shape).astype(np.float32) * pos_std * 0.005
    tgt_pos = tgt_pos + np.random.randn(*tgt_pos.shape).astype(np.float32) * pos_std * 0.005

    return ctx_pos, ctx_dt, ctx_vel, tgt_pos, tgt_vel


def _jitter_sampling(
    ctx_pos: np.ndarray,
    ctx_t: np.ndarray,
    ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    不规则采样增强: 抖动历史时间戳后在原轨迹上线性重采样。

    公开数据集是固定 0.1 s 采样，而 KinFlow 的核心卖点是不规则采样鲁棒性；
    若不施加时间抖动，编码器的 Δt 特征与连续时间编码只会见到单一间隔。
    首尾时刻保持不变（窗口边界不动），中间点抖动幅度为 ±ratio × 名义 dt，
    并强制时间戳严格递增，位置按扰动后的时刻线性插值。

    Returns:
        ctx_pos_new, ctx_t_new, ctx_dt_new
    """
    n = len(ctx_t)
    if n < 3:
        return ctx_pos, ctx_t, np.concatenate([[0.0], np.diff(ctx_t)]).astype(np.float32)

    nominal = float(np.median(np.diff(ctx_t)))
    if nominal <= 0:
        return ctx_pos, ctx_t, np.concatenate([[0.0], np.diff(ctx_t)]).astype(np.float32)

    offsets = np.random.uniform(-ratio, ratio, n) * nominal
    offsets[0] = 0.0
    offsets[-1] = 0.0
    t_new = ctx_t + offsets.astype(np.float32)

    # 强制严格递增（最小间隔为名义间隔的 20%）
    min_gap = 0.2 * nominal
    for i in range(1, n):
        if t_new[i] - t_new[i - 1] < min_gap:
            t_new[i] = t_new[i - 1] + min_gap
    if t_new[-1] <= t_new[-2]:
        t_new[-1] = t_new[-2] + min_gap

    pos_new = np.stack(
        [np.interp(t_new, ctx_t, ctx_pos[:, axis]) for axis in range(3)], axis=1,
    ).astype(np.float32)

    dt_new = np.zeros(n, dtype=np.float32)
    dt_new[1:] = t_new[1:] - t_new[:-1]
    return pos_new, t_new.astype(np.float32), dt_new


class KinFlowWindowDataset(Dataset):
    """KinFlow 窗口数据集：滑动窗口 + 增强 + 逐轨迹标准化。"""

    def __init__(
        self,
        trajectories: list[tuple[np.ndarray, np.ndarray]],
        ctx_len: int = 20,
        tgt_len: int = 10,
        normalize: bool = True,
        augment: bool = False,
        max_windows: int = None,
        seed: int = 42,
        time_jitter: float = 0.0,
    ):
        self.ctx_len = ctx_len
        self.tgt_len = tgt_len
        self.normalize = normalize
        self.augment = augment
        self.time_jitter = float(time_jitter)

        windows: list[dict] = []
        for pos, ts in trajectories:
            windows.extend(build_sliding_windows(pos, ts, ctx_len, tgt_len))

        if max_windows and len(windows) > max_windows:
            rng = default_rng(seed)
            keep = rng.choice(len(windows), size=max_windows, replace=False)
            windows = [windows[i] for i in sorted(keep.tolist())]

        self.windows = windows
        print(f"[Dataset] {len(trajectories)} 条轨迹 → {len(self.windows)} 个训练窗口")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        w = self.windows[idx]

        ctx_pos = w["ctx_pos"].astype(np.float32).copy()
        ctx_t = w["ctx_t"].astype(np.float32).copy()
        ctx_dt = w["ctx_dt"].astype(np.float32).copy()
        ctx_vel = w["ctx_vel"].astype(np.float32).copy()
        tgt_pos = w["tgt_pos"].astype(np.float32).copy()
        tgt_t = w["tgt_t"].astype(np.float32).copy()
        tgt_vel = w["tgt_vel"].astype(np.float32).copy()

        if self.augment:
            ctx_pos, ctx_dt, ctx_vel, tgt_pos, tgt_vel = _augment(
                ctx_pos, ctx_t, ctx_dt, ctx_vel, tgt_pos, tgt_vel,
            )
            if self.time_jitter > 0:
                ctx_pos, ctx_t, ctx_dt = _jitter_sampling(ctx_pos, ctx_t, self.time_jitter)
                ctx_vel = estimate_velocity(ctx_pos, ctx_t)

        # 归一化统计量仅由历史窗口给出（推理时同样只有历史）
        p_mean = ctx_pos.mean(axis=0)
        p_std = np.maximum(ctx_pos.std(axis=0), POSITION_STD_FLOOR)

        if self.normalize:
            ctx_pos = (ctx_pos - p_mean) / p_std
            tgt_pos = (tgt_pos - p_mean) / p_std
            ctx_vel = ctx_vel / p_std
            tgt_vel = tgt_vel / p_std

        return {
            "ctx_pos": torch.from_numpy(ctx_pos).float(),
            "ctx_t": torch.from_numpy(ctx_t).float(),
            "ctx_dt": torch.from_numpy(ctx_dt).float(),
            "ctx_vel": torch.from_numpy(ctx_vel).float(),
            "tgt_pos": torch.from_numpy(tgt_pos).float(),
            "tgt_t": torch.from_numpy(tgt_t).float(),
            "tgt_vel": torch.from_numpy(tgt_vel).float(),
            "p_mean": torch.from_numpy(p_mean).float(),
            "p_std": torch.from_numpy(p_std).float(),
        }


def kinflow_collate(batch: list[dict]) -> dict:
    """固定长度窗口，直接堆叠成批次。"""
    return {key: torch.stack([item[key] for item in batch], dim=0) for key in batch[0]}


__all__ = [
    "load_dat_trajectory",
    "load_trajectory_set",
    "build_sliding_windows",
    "KinFlowWindowDataset",
    "kinflow_collate",
    "_jitter_sampling",
]
