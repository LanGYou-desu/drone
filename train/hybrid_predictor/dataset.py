"""
轨迹数据集 — 滑动窗口采样 + 标准化

支持:
  - 从 .dat 文件加载真实轨迹
  - 从 .npz 文件加载合成轨迹
  - 滑动窗口采样 (context, target) 对
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


def estimate_velocity(positions: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """
    用中心差分 + 端点前向/后向差分估计各点的速度向量。

    内部点用中心差分（O(Δt²) 精度），端点用单侧差分（O(Δt) 精度），
    避免简单复制相邻值导致的边界速度失真。

    Args:
        positions: (N, 3) 位置序列
        timestamps: (N,) 时间戳

    Returns:
        velocities: (N, 3) 速度估计
    """
    N = len(positions)
    vel = np.zeros((N, 3), dtype=np.float32)
    if N < 2:
        return vel

    if N >= 3:
        # 内部点: 中心差分
        for i in range(1, N - 1):
            dt_span = timestamps[i + 1] - timestamps[i - 1]
            if dt_span > 0:
                vel[i] = (positions[i + 1] - positions[i - 1]) / dt_span
        # 起点: 前向差分
        dt_fwd = timestamps[1] - timestamps[0]
        if dt_fwd > 0:
            vel[0] = (positions[1] - positions[0]) / dt_fwd
        # 终点: 后向差分
        dt_bwd = timestamps[-1] - timestamps[-2]
        if dt_bwd > 0:
            vel[-1] = (positions[-1] - positions[-2]) / dt_bwd
    else:  # N == 2
        dt_span = timestamps[1] - timestamps[0]
        if dt_span > 0:
            v = (positions[1] - positions[0]) / dt_span
            vel[0] = vel[1] = v
    return vel


def load_trajectory_from_dat(file_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    从 .dat 文件加载轨迹（x y z t 格式）。

    Returns:
        positions: (N, 3), timestamps: (N,)
    """
    points = []
    timestamps = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 4:
                x, y, z, t = map(float, parts[:4])
                points.append([x, y, z])
                timestamps.append(t)

    if len(points) < 3:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    return np.array(points, dtype=np.float32), np.array(timestamps, dtype=np.float32)


def load_all_trajectories(
    data_dir: str = None,
    synthetic_dir: str = None,
    max_files: int = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    加载所有可用轨迹。

    Args:
        data_dir:     真实 .dat 文件目录（可选，默认不使用运行时数据）
        synthetic_dir: 合成 .npz 目录（如 dataset/train/）
        max_files:    最大文件数

    Returns:
        [(positions, timestamps), ...]
    """
    trajectories = []

    # 加载真实运行时数据（可选，默认不使用）
    if data_dir and os.path.isdir(data_dir):
        for fname in sorted(os.listdir(data_dir)):
            if fname.endswith('.dat'):
                pos, ts = load_trajectory_from_dat(os.path.join(data_dir, fname))
                if len(pos) >= 5:
                    trajectories.append((pos, ts))

    # 加载合成训练数据（.npz 或 .dat 格式）
    if synthetic_dir and os.path.isdir(synthetic_dir):
        for fname in sorted(os.listdir(synthetic_dir)):
            fpath = os.path.join(synthetic_dir, fname)
            if fname.endswith('.npz'):
                data = np.load(fpath)
                pos = data['positions']
                ts = data['timestamps']
                if len(pos) >= 5:
                    trajectories.append((pos, ts))
            elif fname.endswith('.dat'):
                # 用户自行导入的 .dat 格式数据（x y z t）
                pos, ts = load_trajectory_from_dat(fpath)
                if len(pos) >= 5:
                    trajectories.append((pos, ts))

    if max_files and len(trajectories) > max_files:
        trajectories = trajectories[:max_files]

    return trajectories


def sliding_window_sample(
    positions: np.ndarray,
    timestamps: np.ndarray,
    ctx_len: int = 20,
    tgt_len: int = 10,
) -> list[dict]:
    """
    从单条轨迹中滑动窗口提取 (上下文, 目标) 对。

    Returns:
        [{ctx_pos, ctx_t, ctx_dt, ctx_vel, tgt_pos, tgt_t, tgt_vel}, ...]
    """
    N = len(positions)
    min_len = ctx_len + tgt_len
    if N < min_len:
        return []

    # 预计算整条轨迹的归一化统计量（与推理时一致，避免 train/inference 分布不匹配）
    traj_pos = positions
    traj_mean = traj_pos.mean(axis=0)
    traj_std = traj_pos.std(axis=0).clip(min=1e-3)

    samples = []
    stride = max(1, min(ctx_len // 2, tgt_len // 2))

    for start in range(0, N - min_len + 1, stride):
        end_ctx = start + ctx_len
        end_tgt = end_ctx + tgt_len

        ctx_pos = positions[start:end_ctx]
        ctx_t = timestamps[start:end_ctx]
        ctx_dt = np.zeros(ctx_len, dtype=np.float32)
        ctx_dt[1:] = ctx_t[1:] - ctx_t[:-1]

        # 估计速度（中心差分）
        ctx_vel = estimate_velocity(ctx_pos, ctx_t)

        tgt_pos = positions[end_ctx:end_tgt]
        tgt_t = timestamps[end_ctx:end_tgt]

        tgt_vel = estimate_velocity(tgt_pos, tgt_t)

        samples.append({
            "ctx_pos": ctx_pos,
            "ctx_t": ctx_t,
            "ctx_dt": ctx_dt,
            "ctx_vel": ctx_vel,
            "tgt_pos": tgt_pos,
            "tgt_t": tgt_t,
            "tgt_vel": tgt_vel,
            "traj_mean": traj_mean,
            "traj_std": traj_std,
        })

    return samples


class TrajectoryDataset(Dataset):
    """
    PyTorch 轨迹数据集。

    对每条轨迹做滑动窗口采样，标准化后按批次返回。
    """

    def __init__(
        self,
        trajectories: list[tuple[np.ndarray, np.ndarray]],
        ctx_len: int = 20,
        tgt_len: int = 10,
        normalize: bool = True,
        augment: bool = False,
    ):
        self.ctx_len = ctx_len
        self.tgt_len = tgt_len
        self.normalize = normalize
        self.augment = augment
        self.samples: list[dict] = []

        for pos, ts in trajectories:
            self.samples.extend(sliding_window_sample(pos, ts, ctx_len, tgt_len))

        print(f"[Dataset] {len(trajectories)} 条轨迹 → {len(self.samples)} 个训练窗口")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        ctx_pos = sample["ctx_pos"].copy()
        ctx_t = sample["ctx_t"].copy()
        ctx_dt = sample["ctx_dt"].copy()
        ctx_vel = sample["ctx_vel"].copy()
        tgt_pos = sample["tgt_pos"].copy()
        tgt_t = sample["tgt_t"].copy()
        tgt_vel = sample["tgt_vel"].copy()

        # 随机数据增强
        if self.augment:
            # 检测垂直运动占比，爬升/下降为主的轨迹不进行 Y 轴旋转
            # (绕Y轴旋转会改变水平方向但对高度无影响，对纯爬升无意义)
            all_pos = np.concatenate([ctx_pos, tgt_pos], axis=0)
            vertical_range = np.ptp(all_pos[:, 1])  # Y 轴跨度
            horizontal_range = max(np.ptp(all_pos[:, 0]), np.ptp(all_pos[:, 2]), 1e-6)
            is_mostly_horizontal = vertical_range < 3.0 * horizontal_range

            if is_mostly_horizontal:
                # 随机旋转（绕 Y 轴，仅在水平运动为主时有效）
                angle = np.random.uniform(0, 2 * np.pi)
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                R = np.array([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]])
                ctx_pos = ctx_pos @ R.T
                ctx_vel = ctx_vel @ R.T
                tgt_pos = tgt_pos @ R.T
                tgt_vel = tgt_vel @ R.T

            # 随机缩放（±10%）
            scale = np.random.uniform(0.9, 1.1)
            ctx_pos *= scale
            tgt_pos *= scale
            ctx_vel *= scale
            tgt_vel *= scale

            # 小幅随机加噪
            noise_scale = 0.005 * np.std(all_pos, axis=0).clip(min=1e-3)
            ctx_pos += np.random.randn(*ctx_pos.shape) * noise_scale
            tgt_pos += np.random.randn(*tgt_pos.shape) * noise_scale

        # 标准化：使用整条轨迹的统计量（与推理时一致，避免分布不匹配）
        if self.normalize:
            p_mean = sample.get("traj_mean", ctx_pos.mean(axis=0))
            p_std = sample.get("traj_std", ctx_pos.std(axis=0).clip(min=1e-3))
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
        }


def collate_fn(batch: list[dict]) -> dict:
    """批次整理：各样本可能有不同长度，本实现固定 ctx_len/tgt_len，直接堆叠"""
    result = {}
    for key in batch[0].keys():
        result[key] = torch.stack([item[key] for item in batch], dim=0)
    return result
