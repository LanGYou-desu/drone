"""
三维轨迹重建 — 异常点剔除 + 缺测补全 + 加权平滑 + 质量评估

经时间同步与多传感器融合后得到的轨迹仍可能存在:
  · 单传感器时刻的测量跳点（误检、遮挡恢复瞬间的抖动）；
  · 短时丢帧造成的空洞；
  · 融合不确定度随时刻变化（不同数量的传感器参与）。

重建流程:
  1. 稳健异常点检测: 局地中值残差 + MAD 鲁棒阈值；
  2. 短空洞补全: 相邻有效点间做 Catmull-Rom 三次插值（保持 C¹ 连续），
     超过 max_gap_s 的空洞不补，保留为断段；
  3. 加权平滑: 三角核 + 1/σ² 权重的滑动平均，端点保持；
  4. 运动学: 速度、加速度、曲率；
  5. 质量: 拟合残差、噪声估计、补洞/剔除数量、连续性、平均不确定度、逐点置信度。

纯 numpy 实现。
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ReconstructionConfig:
    """重建参数。"""

    max_gap_s: float = 0.8          # 允许补全的最长空洞（秒）
    smooth_window: int = 5          # 平滑窗口（奇数，<=1 表示不平滑）
    outlier_sigma: float = 3.5      # 鲁棒阈值倍数
    min_confidence: float = 0.05
    sigma_scale: float = 5.0        # 不确定度到置信度的映射尺度


@dataclass
class ReconstructionResult:
    """重建结果。"""

    times: np.ndarray                 # (M,)
    raw: np.ndarray                   # (M,3) 融合原始
    points: np.ndarray                # (M,3) 重建后
    filled: np.ndarray                # (M,) bool 补洞点
    outliers: np.ndarray              # (M,) bool 被剔除的异常点
    confidence: np.ndarray            # (M,) 0~1
    sigmas: np.ndarray                # (M,) 不确定度
    speeds: np.ndarray                # (M,)
    accelerations: np.ndarray
    curvatures: np.ndarray
    segments: list[list[int]]         # 连续有效段 [start, end]
    metrics: dict = field(default_factory=dict)

    def to_dict(self, max_points: int = None) -> dict:
        m = len(self.times)
        idx = list(range(m))
        if max_points and m > max_points:
            idx = np.linspace(0, m - 1, max_points).astype(int).tolist()
        return {
            'times': [round(float(self.times[i]), 4) for i in idx],
            'raw': [[round(float(v), 4) for v in self.raw[i]] for i in idx],
            'points': [[round(float(v), 4) for v in self.points[i]] for i in idx],
            'filled': [bool(self.filled[i]) for i in idx],
            'outliers': [bool(self.outliers[i]) for i in idx],
            'confidence': [round(float(self.confidence[i]), 4) for i in idx],
            'sigmas': [round(float(self.sigmas[i]), 4) for i in idx],
            'speeds': [round(float(self.speeds[i]), 4) if np.isfinite(self.speeds[i]) else None for i in idx],
            'accelerations': [round(float(self.accelerations[i]), 4)
                              if np.isfinite(self.accelerations[i]) else None for i in idx],
            'curvatures': [round(float(self.curvatures[i]), 6)
                           if np.isfinite(self.curvatures[i]) else None for i in idx],
            'segments': self.segments,
            'metrics': self.metrics,
        }


# ── 内部工具 ──────────────────────────────────────────

def _local_median_residual(points: np.ndarray, window: int = 5) -> np.ndarray:
    """逐点与局地中值轨迹的偏差（NaN 安全）。"""
    m = len(points)
    resid = np.full(m, np.nan)
    half = max(window // 2, 1)
    for i in range(m):
        lo, hi = max(0, i - half), min(m, i + half + 1)
        chunk = points[lo:hi]
        chunk = chunk[np.isfinite(chunk).all(axis=1)]
        if len(chunk) >= 2:
            resid[i] = np.linalg.norm(points[i] - np.median(chunk, axis=0))
    return resid


def _catmull_rom(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray,
                 s: np.ndarray) -> np.ndarray:
    """Catmull-Rom 三次插值，s ∈ [0,1] 位于 p1 与 p2 之间。"""
    s = s[:, None]
    s2, s3 = s * s, s * s * s
    return 0.5 * ((2 * p1)
                  + (-p0 + p2) * s
                  + (2 * p0 - 5 * p1 + 4 * p2 - p3) * s2
                  + (-p0 + 3 * p1 - 3 * p2 + p3) * s3)


def _fill_gaps(times: np.ndarray, points: np.ndarray, max_gap_s: float) -> tuple[np.ndarray, np.ndarray]:
    """对长度不超过 max_gap_s 的内部空洞做 Catmull-Rom 补全。"""
    out = points.copy()
    filled = np.zeros(len(times), dtype=bool)
    valid_idx = np.where(np.isfinite(points).all(axis=1))[0]
    if len(valid_idx) < 3:
        return out, filled

    for left, right in zip(valid_idx[:-1], valid_idx[1:]):
        if right - left <= 1:
            continue
        if times[right] - times[left] > max_gap_s + 1e-9:
            continue
        i0 = max(valid_idx[0], left - 1)
        i3 = min(valid_idx[-1], right + 1)
        p0, p1 = points[i0], points[left]
        p2, p3 = points[right], points[i3]
        span = times[right] - times[left]
        for k in range(left + 1, right):
            s = np.array([(times[k] - times[left]) / max(span, 1e-9)])
            out[k] = _catmull_rom(p0, p1, p2, p3, s)[0]
            filled[k] = True
    return out, filled


def _smooth(times: np.ndarray, points: np.ndarray, sigmas: np.ndarray,
            window: int) -> np.ndarray:
    """三角核 + 逆方差加权的滑动平均（不跨断段，端点保持）。"""
    m = len(points)
    if window <= 1 or m < 3:
        return points.copy()

    half = window // 2
    out = points.copy()
    finite = np.isfinite(points).all(axis=1)
    for i in range(m):
        if not finite[i]:
            continue
        lo, hi = max(0, i - half), min(m, i + half + 1)
        idx = [k for k in range(lo, hi) if finite[k]]
        if len(idx) < 2:
            continue
        w = np.array([
            (1.0 - abs(k - i) / (half + 1.0)) / max(sigmas[k] if np.isfinite(sigmas[k]) else 1.0, 1e-3) ** 2
            for k in idx
        ])
        out[i] = (w[:, None] * points[idx]).sum(axis=0) / w.sum()
    return out


def _kinematics(times: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """速度、加速度、曲率（NaN 安全）。"""
    m = len(points)
    speed = np.full(m, np.nan)
    accel = np.full(m, np.nan)
    curv = np.full(m, np.nan)
    valid = np.isfinite(points).all(axis=1)
    if valid.sum() < 2:
        return speed, accel, curv

    for i in range(1, m):
        if valid[i] and valid[i - 1]:
            dt = times[i] - times[i - 1]
            if dt > 1e-9:
                speed[i] = np.linalg.norm(points[i] - points[i - 1]) / dt
    if valid[0]:
        speed[0] = speed[1] if np.isfinite(speed[1]) else np.nan

    for i in range(1, m):
        if np.isfinite(speed[i]) and np.isfinite(speed[i - 1]):
            dt = times[i] - times[i - 1]
            if dt > 1e-9:
                accel[i] = (speed[i] - speed[i - 1]) / dt

    for i in range(1, m - 1):
        if valid[i - 1] and valid[i] and valid[i + 1]:
            d1 = points[i] - points[i - 1]
            d2 = points[i + 1] - points[i]
            cr = np.cross(d1, d2)
            n1 = np.linalg.norm(d1)
            curv[i] = float(np.linalg.norm(cr) / (n1 ** 3)) if n1 > 1e-6 else 0.0
    return speed, accel, curv


def _segments(points: np.ndarray) -> list[list[int]]:
    """连续有效点段。"""
    valid = np.isfinite(points).all(axis=1)
    segs, start = [], None
    for i, ok in enumerate(valid):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            segs.append([start, i - 1])
            start = None
    if start is not None:
        segs.append([start, len(valid) - 1])
    return segs


# ── 主入口 ────────────────────────────────────────────

def reconstruct(
    times: np.ndarray,
    positions: np.ndarray,
    sigmas: np.ndarray,
    config: ReconstructionConfig = None,
) -> ReconstructionResult:
    """
    由融合轨迹重建平滑、连续、带置信度的三维轨迹。

    Args:
        times:     主时间栅格 (M,)
        positions: 融合位置 (M,3)，缺测 NaN
        sigmas:    融合不确定度 (M,)，缺测 NaN
        config:    重建参数

    Returns:
        ReconstructionResult
    """
    cfg = config or ReconstructionConfig()
    times = np.asarray(times, dtype=np.float64)
    raw = np.asarray(positions, dtype=np.float64).copy()
    sig = np.asarray(sigmas, dtype=np.float64).copy()
    m = len(times)

    outliers = np.zeros(m, dtype=bool)
    if m == 0:
        return ReconstructionResult(times, raw, raw.copy(), np.zeros(0, bool),
                                    outliers, np.zeros(0), sig,
                                    np.zeros(0), np.zeros(0), np.zeros(0), [], {})

    # 1) 稳健异常点剔除
    resid = _local_median_residual(raw)
    finite_resid = resid[np.isfinite(resid)]
    if len(finite_resid) >= 4:
        med = float(np.median(finite_resid))
        mad = float(np.median(np.abs(finite_resid - med))) * 1.4826
        thr = med + max(cfg.outlier_sigma * mad, 1e-6)
        noise_floor = float(np.nanmedian(np.where(np.isfinite(sig), sig, np.nan))) if np.isfinite(sig).any() else 0.5
        thr = max(thr, 0.8 * noise_floor)
        outliers = np.isfinite(resid) & (resid > thr)
        if outliers.sum() > m * 0.4:      # 过度剔除时放弃剔除
            outliers[:] = False
    cleaned = raw.copy()
    cleaned[outliers] = np.nan

    # 2) 空洞补全
    filled_points, filled = _fill_gaps(times, cleaned, cfg.max_gap_s)
    sig_filled = sig.copy()
    if filled.any():
        med_sig = float(np.nanmedian(sig)) if np.isfinite(sig).any() else 1.0
        sig_filled[filled] = np.where(np.isfinite(sig_filled[filled]), sig_filled[filled], med_sig * 1.5)

    # 3) 平滑
    points = _smooth(times, filled_points, sig_filled, cfg.smooth_window)

    # 4) 运动学
    speed, accel, curv = _kinematics(times, points)

    # 5) 置信度: 不确定度越小越可信；补洞点打折
    conf = np.zeros(m)
    finite = np.isfinite(points).all(axis=1)
    if np.isfinite(sig_filled).any():
        s = np.where(np.isfinite(sig_filled), sig_filled, np.nanmax(sig_filled))
        conf[finite] = np.exp(-s[finite] / max(cfg.sigma_scale, 1e-6))
    else:
        conf[finite] = 0.5
    conf[filled] *= 0.6
    conf = np.clip(conf, cfg.min_confidence, 1.0)

    # 6) 质量指标
    obs = np.isfinite(raw).all(axis=1)
    fit_resid = np.linalg.norm(points[obs] - raw[obs], axis=1) if obs.any() else np.zeros(0)
    noise = float(np.sqrt(np.mean(fit_resid ** 2))) if len(fit_resid) else 0.0
    dt = float(np.median(np.diff(times))) if m > 1 else 0.0
    dv = np.abs(np.diff(speed))
    metrics = {
        'n_master': int(m),
        'hz': round(1.0 / dt, 3) if dt > 0 else 0.0,
        'duration': round(float(times[-1] - times[0]), 3),
        'observed_ratio': round(float(obs.mean()), 4),
        'reconstructed_ratio': round(float(finite.mean()), 4),
        'n_outliers': int(outliers.sum()),
        'n_filled': int(filled.sum()),
        'noise_estimate': round(noise, 4),
        'mean_sigma': round(float(np.nanmean(sig_filled)), 4) if np.isfinite(sig_filled).any() else None,
        'max_speed': round(float(np.nanmax(speed)), 3) if np.isfinite(speed).any() else 0.0,
        'max_accel': round(float(np.nanmax(np.abs(accel))), 3) if np.isfinite(accel).any() else 0.0,
        'max_speed_jump': round(float(np.nanmax(dv)), 3) if len(dv) and np.isfinite(dv).any() else 0.0,
        'mean_confidence': round(float(np.mean(conf[finite])), 4) if finite.any() else 0.0,
        'segments': len(_segments(points)),
        'path_length': round(float(np.nansum(np.linalg.norm(
            np.diff(points[np.isfinite(points).all(axis=1)], axis=0), axis=1))), 3)
        if finite.sum() >= 2 else 0.0,
    }

    return ReconstructionResult(
        times=times, raw=raw, points=points, filled=filled, outliers=outliers,
        confidence=conf, sigmas=sig_filled, speeds=speed, accelerations=accel,
        curvatures=curv, segments=_segments(points), metrics=metrics,
    )


__all__ = ['ReconstructionConfig', 'ReconstructionResult', 'reconstruct']
