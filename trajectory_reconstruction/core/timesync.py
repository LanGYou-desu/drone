"""
多平台时间步同步 — 时钟偏置估计 + 统一时间栅格重采样

无人机监测中，可见光/红外/雷达的采样率、起始时刻、丢帧情况各不相同；
若直接按数组下标混合，会引入数十到数百毫秒的错位，融合后的轨迹出现"锯齿"或
虚假的高加速度。本模块负责:

  1. 以运动能量（速率剖面）互相关估计各传感器相对参考传感器的时钟偏置 delta；
  2. 按 delta 平移时间轴后，把每条轨迹重采样到统一的主时间栅格（master timeline）；
  3. 标记越界与超时丢帧（gap），并给出覆盖率、最大空档、采样抖动、
     与多传感器共识轨迹的残差等同步质量指标。

纯 numpy 实现，无 Flask/全局状态依赖，便于单测与离线评估。
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SensorSeries:
    """单个传感器的一条时序轨迹。"""

    sensor_id: str
    times: np.ndarray            # (N,) 秒
    positions: np.ndarray        # (N, 3) 米
    name: str = ''
    color: str = '#999999'
    sigma: float = 1.0           # 先验测量噪声（米），用于加权融合

    def clean(self) -> 'SensorSeries':
        """按时间排序、去重、剔除非有限值。"""
        t = np.asarray(self.times, dtype=np.float64).reshape(-1)
        p = np.asarray(self.positions, dtype=np.float64)
        if p.ndim != 2 or p.shape[1] != 3:
            raise ValueError(f'{self.sensor_id}: positions 必须为 (N,3)')
        n = min(len(t), len(p))
        t, p = t[:n], p[:n]
        mask = np.isfinite(t) & np.isfinite(p).all(axis=1)
        t, p = t[mask], p[mask]
        order = np.argsort(t, kind='stable')
        t, p = t[order], p[order]
        keep = np.concatenate([[True], np.diff(t) > 1e-9]) if len(t) else np.zeros(0, bool)
        return SensorSeries(self.sensor_id, t[keep], p[keep],
                            self.name, self.color, self.sigma)


@dataclass
class SyncMetrics:
    """同步质量指标（对标申报书"数据同步"考核点）。"""

    hz: float
    dt: float
    master_start: float
    master_end: float
    n_master: int
    offsets: dict[str, float] = field(default_factory=dict)       # 估计的时钟偏置（秒）
    correlations: dict[str, float] = field(default_factory=dict)  # 互相关峰值
    offset_confidence: dict[str, float] = field(default_factory=dict)  # 偏置估计置信度 0~1
    drift: dict[str, float] = field(default_factory=dict)         # 前后半段偏置差 / 半段时长
    coverage: dict[str, float] = field(default_factory=dict)      # 主栅格覆盖比例
    max_gap: dict[str, float] = field(default_factory=dict)       # 最大连续丢帧（秒）
    jitter: dict[str, float] = field(default_factory=dict)        # 原始采样间隔标准差（秒）
    residual: dict[str, float] = field(default_factory=dict)      # 与多传感器共识位置的残差（米）
    n_points: dict[str, int] = field(default_factory=dict)
    simulated_offsets: dict[str, float] = field(default_factory=dict)
    recovery_error: dict[str, float] = field(default_factory=dict)  # 模拟偏移下的恢复误差（秒）
    reference_id: str = ''

    def to_dict(self) -> dict:
        return {
            'hz': round(self.hz, 4), 'dt': round(self.dt, 6),
            'master_start': round(self.master_start, 4),
            'master_end': round(self.master_end, 4),
            'n_master': self.n_master,
            'offsets': {k: round(v, 4) for k, v in self.offsets.items()},
            'correlations': {k: round(v, 4) for k, v in self.correlations.items()},
            'offset_confidence': {k: round(v, 4) for k, v in self.offset_confidence.items()},
            'drift': {k: round(v, 6) for k, v in self.drift.items()},
            'coverage': {k: round(v, 4) for k, v in self.coverage.items()},
            'max_gap': {k: round(v, 4) for k, v in self.max_gap.items()},
            'jitter': {k: round(v, 6) for k, v in self.jitter.items()},
            'residual': {k: round(v, 4) for k, v in self.residual.items()},
            'n_points': dict(self.n_points),
            'simulated_offsets': {k: round(v, 4) for k, v in self.simulated_offsets.items()},
            'recovery_error': {k: round(v, 4) for k, v in self.recovery_error.items()},
            'reference_id': self.reference_id,
        }


@dataclass
class SyncResult:
    """同步结果: 统一时间栅格 + 各传感器对齐后的位置（缺测为 NaN）。"""

    times: np.ndarray                       # (M,)
    aligned: dict[str, np.ndarray]          # sensor_id -> (M,3)
    present: dict[str, np.ndarray]          # sensor_id -> (M,) bool
    metrics: SyncMetrics

    def to_dict(self, max_points: int = None) -> dict:
        m = max_points or len(self.times)
        idx = np.linspace(0, len(self.times) - 1, min(m, len(self.times))).astype(int) if len(self.times) else []
        sensors = {}
        for sid, arr in self.aligned.items():
            sensors[sid] = [
                [None, None, None] if not np.isfinite(arr[i]).all() else [round(float(v), 4) for v in arr[i]]
                for i in idx
            ]
        return {
            'times': [round(float(self.times[i]), 4) for i in idx],
            'sensors': sensors,
            'metrics': self.metrics.to_dict(),
        }


# ── 内部工具 ──────────────────────────────────────────

def _speed_profile(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    """逐点速率（中心差分，端点单侧差分），单位 m/s。"""
    n = len(t)
    speed = np.zeros(n)
    if n < 2:
        return speed
    dt = np.diff(t)
    dt[dt <= 1e-9] = 1e-9
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1) / dt
    if n == 2:
        speed[:] = seg[0]
        return speed
    speed[0] = seg[0]
    speed[-1] = seg[-1]
    speed[1:-1] = (seg[:-1] + seg[1:]) / 2.0
    return speed


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 4:
        return 0.0
    x, y = a[mask], b[mask]
    x = x - x.mean()
    y = y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / denom) if denom > 1e-9 else 0.0


@dataclass
class OffsetEstimate:
    """单次时钟偏置估计结果。"""

    delta: float = 0.0          # 估计偏置（秒）
    correlation: float = 0.0    # 最优滞后处的相关系数
    confidence: float = 0.0     # 0~1，相关峰的显著度
    prominence: float = 0.0     # 最优相关与中位相关的差

    def __iter__(self):
        """兼容旧的 (delta, correlation) 解包写法。"""
        return iter((self.delta, self.correlation))


def _smooth_positions(pos: np.ndarray, window: int) -> np.ndarray:
    """沿时间轴做滑动平均（边缘复制），抑制测量噪声。"""
    if window <= 1 or len(pos) < window:
        return pos.copy()
    kernel = np.ones(window) / window
    out = np.empty_like(pos)
    for axis in range(pos.shape[1]):
        pad = np.pad(pos[:, axis], (window // 2, window // 2), mode='edge')
        out[:, axis] = np.convolve(pad, kernel, mode='valid')
    return out


def estimate_clock_offset(ref: SensorSeries, sensor: SensorSeries,
                          max_offset: float = 0.6,
                          rate: float = 10.0,
                          smooth_window: int = 5) -> OffsetEstimate:
    """
    用三维位置轨迹的归一化互相关估计时钟偏置（粗搜 + 精搜）。

    设计要点:
      1. 用位置而不是速率作相关信号 —— 位置差分会把噪声放大 1/Δt 倍；
      2. 只在高动态段（速率高于中位数）做相关 —— 匀速直线段几乎不携带时间信息；
      3. 位置先做 0.5 s 平滑，进一步压制噪声；
      4. 粗搜 0.1 s 步长，再在最优点 ±0.1 s 内以 0.02 s 精搜；
      5. 以"相峰显著度"给出置信度：轨迹机动性不足时相关系数曲线平坦，
         此时不强行平移（delta=0），避免错误对齐。

    Returns:
        OffsetEstimate
    """
    t0 = max(ref.times[0], sensor.times[0])
    t1 = min(ref.times[-1], sensor.times[-1])
    if t1 - t0 < max(0.5, 6.0 / rate):
        return OffsetEstimate()

    grid = np.arange(t0, t1, 1.0 / rate)
    if len(grid) < 10:
        return OffsetEstimate()

    ref_pos = np.stack([np.interp(grid, ref.times, ref.positions[:, axis])
                        for axis in range(3)], axis=1)
    ref_smooth = _smooth_positions(ref_pos, smooth_window)

    # 高动态段掩码（转弯/加减速处才有时间信息）
    vel = np.gradient(ref_pos, 1.0 / rate, axis=0)
    speed = np.linalg.norm(vel, axis=1)
    threshold = np.percentile(speed, 50)
    mask = speed >= threshold
    if mask.sum() < 8:
        mask = np.ones(len(grid), dtype=bool)

    def _score(delta: float):
        raw = np.stack([
            np.interp(grid + delta, sensor.times, sensor.positions[:, axis],
                      left=np.nan, right=np.nan) for axis in range(3)
        ], axis=1)
        idx = np.where(np.isfinite(raw[:, 0]))[0]
        if len(idx) < 8:
            return None
        # 有效段内先插值再平滑，避免 NaN 污染
        filled = np.stack([np.interp(grid, grid[idx], raw[idx, axis]) for axis in range(3)], axis=1)
        smooth = _smooth_positions(filled, smooth_window)
        corrs = []
        for axis in range(3):
            x, y = ref_smooth[mask, axis], smooth[mask, axis]
            if np.std(x) < 1e-9 or np.std(y) < 1e-9:
                continue
            corrs.append(_pearson(x, y))
        if not corrs:
            return None
        return float(np.mean(corrs))

    step = 1.0 / rate
    n_step = int(round(max_offset / step))
    curve = {}
    for k in range(-n_step, n_step + 1):
        delta = round(k * step, 6)
        sc = _score(delta)
        if sc is not None:
            curve[delta] = sc
    if not curve:
        return OffsetEstimate()

    # 轻微偏好小偏置（先验: 大多数时钟近似同步），避免平台期任意漂移
    def _objective(corr, delta):
        return corr - 0.02 * abs(delta) / max(max_offset, 1e-6)

    best_delta = max(curve, key=lambda d: _objective(curve[d], d))
    best_corr = curve[best_delta]

    # 精搜
    for delta in np.arange(best_delta - step, best_delta + step + 1e-9, 0.02):
        delta = float(np.clip(round(float(delta), 4), -max_offset, max_offset))
        sc = _score(delta)
        if sc is not None and _objective(sc, delta) > _objective(best_corr, best_delta):
            best_delta, best_corr = delta, sc

    median_corr = float(np.median(list(curve.values())))
    prominence = best_corr - median_corr
    confidence = float(np.clip(prominence / 0.15, 0.0, 1.0)
                       * np.clip((best_corr - 0.3) / 0.4, 0.0, 1.0))

    # 相关峰不显著 -> 认为无可靠偏置
    if best_corr < 0.3 or prominence < 0.03:
        return OffsetEstimate(0.0, max(best_corr, 0.0), confidence, prominence)
    return OffsetEstimate(best_delta, best_corr, confidence, prominence)


def _resample(t: np.ndarray, p: np.ndarray, grid: np.ndarray,
              max_sample_gap: float) -> np.ndarray:
    """重采样到 grid；越界或距最近样本超过 max_sample_gap 的位置记为 NaN。"""
    out = np.full((len(grid), 3), np.nan)
    if len(t) < 2:
        return out

    inside = (grid >= t[0] - 1e-9) & (grid <= t[-1] + 1e-9)
    if not inside.any():
        return out

    idx = np.searchsorted(t, grid)
    left = np.clip(idx - 1, 0, len(t) - 1)
    right = np.clip(idx, 0, len(t) - 1)
    nearest_gap = np.minimum(np.abs(grid - t[left]), np.abs(grid - t[right]))
    valid = inside & (nearest_gap <= max_sample_gap)

    for axis in range(3):
        out[valid, axis] = np.interp(grid[valid], t, p[:, axis])
    return out


def _longest_gap(present: np.ndarray, dt: float) -> float:
    """最长连续缺测时长（秒）。"""
    longest = current = 0
    for ok in present:
        if ok:
            longest = max(longest, current)
            current = 0
        else:
            current += 1
    return max(longest, current) * dt


# ── 主入口 ────────────────────────────────────────────

def synchronize(
    series: list[SensorSeries],
    hz: float = 10.0,
    max_offset: float = 0.6,
    reference_id: str = None,
    max_sample_gap: float = 0.5,
    simulate_offsets: dict[str, float] = None,
) -> SyncResult:
    """
    多传感器时间步同步。

    Args:
        series:           各传感器轨迹
        hz:               主时间栅格频率
        max_offset:       时钟偏置搜索范围（秒）
        reference_id:     参考传感器；None 时选采样点最多者
        max_sample_gap:   超过该间隔视为丢帧（秒）
        simulate_offsets: 自检用——在估计前给指定传感器注入已知时钟偏移（秒），
                          用于验证偏置估计器的恢复精度，不影响真实数据。

    Returns:
        SyncResult（统一时间栅格 + 对齐轨迹 + 质量指标）
    """
    cleaned = [s.clean() for s in series]
    cleaned = [s for s in cleaned if len(s.times) >= 2]
    if not cleaned:
        raise ValueError('没有可用的轨迹数据')

    hz = float(max(hz, 1.0))
    dt = 1.0 / hz
    simulate_offsets = dict(simulate_offsets or {})

    # 自检注入: 平移时间轴（等价于传感器时钟超前/滞后）
    working = []
    for s in cleaned:
        shift = float(simulate_offsets.get(s.sensor_id, 0.0))
        working.append(SensorSeries(s.sensor_id, s.times + shift, s.positions,
                                    s.name, s.color, s.sigma))

    if reference_id and any(s.sensor_id == reference_id for s in working):
        ref = next(s for s in working if s.sensor_id == reference_id)
    else:
        ref = max(working, key=lambda s: len(s.times))

    offsets = {ref.sensor_id: 0.0}
    correlations = {ref.sensor_id: 1.0}
    confidences = {ref.sensor_id: 1.0}
    drift = {ref.sensor_id: 0.0}

    for s in working:
        if s.sensor_id == ref.sensor_id:
            continue
        est = estimate_clock_offset(ref, s, max_offset=max_offset)
        offsets[s.sensor_id] = est.delta
        correlations[s.sensor_id] = est.correlation
        confidences[s.sensor_id] = est.confidence

        # 前后半段各估一次，用差值估计漂移率（秒/秒）
        n = len(s.times)
        half = n // 2
        if half >= 8 and len(ref.times) > half:
            d1 = estimate_clock_offset(
                SensorSeries(ref.sensor_id, ref.times[:half + 1], ref.positions[:half + 1]),
                SensorSeries(s.sensor_id, s.times[:half + 1], s.positions[:half + 1]),
                max_offset=max_offset).delta
            d2 = estimate_clock_offset(
                SensorSeries(ref.sensor_id, ref.times[half:], ref.positions[half:]),
                SensorSeries(s.sensor_id, s.times[half:], s.positions[half:]),
                max_offset=max_offset).delta
            span = max(s.times[-1] - s.times[0], 1e-6)
            drift[s.sensor_id] = (d2 - d1) / span
        else:
            drift[s.sensor_id] = 0.0

    # 应用偏置: 平移后的时间轴
    corrected = {}
    for s in working:
        corrected[s.sensor_id] = SensorSeries(
            s.sensor_id, s.times - offsets.get(s.sensor_id, 0.0), s.positions,
            s.name, s.color, s.sigma)

    # 主时间栅格: 取各传感器时间范围的交集，避免边界外推
    start = max(float(c.times[0]) for c in corrected.values())
    end = min(float(c.times[-1]) for c in corrected.values())
    if end - start < dt:
        start = min(float(c.times[0]) for c in corrected.values())
        end = max(float(c.times[-1]) for c in corrected.values())
    n_master = int(np.floor((end - start) / dt)) + 1
    grid = start + np.arange(max(n_master, 1)) * dt

    aligned, present = {}, {}
    coverage, max_gap, jitter, residual, n_points = {}, {}, {}, {}, {}
    for sid, c in corrected.items():
        arr = _resample(c.times, c.positions, grid, max_sample_gap)
        ok = np.isfinite(arr).all(axis=1)
        aligned[sid] = arr
        present[sid] = ok
        coverage[sid] = float(ok.mean()) if len(ok) else 0.0
        max_gap[sid] = _longest_gap(ok, dt)
        jitter[sid] = float(np.std(np.diff(c.times))) if len(c.times) > 2 else 0.0
        n_points[sid] = int(len(c.times))

    # 共识位置（逐时刻中位数）与各传感器残差
    stack = np.full((len(corrected), len(grid), 3), np.nan)
    for i, sid in enumerate(corrected):
        stack[i] = aligned[sid]
    with np.errstate(all='ignore'):
        consensus = np.nanmedian(stack, axis=0)
    for i, sid in enumerate(corrected):
        diff = np.linalg.norm(aligned[sid] - consensus, axis=1)
        valid = np.isfinite(diff)
        residual[sid] = float(np.mean(diff[valid])) if valid.any() else float('nan')

    recovery = {}
    for sid, injected in simulate_offsets.items():
        if sid in offsets:
            recovery[sid] = abs(offsets[sid] - injected)

    metrics = SyncMetrics(
        hz=hz, dt=dt, master_start=float(grid[0]), master_end=float(grid[-1]),
        n_master=len(grid), offsets=offsets, correlations=correlations,
        offset_confidence=confidences, drift=drift,
        coverage=coverage, max_gap=max_gap, jitter=jitter, residual=residual,
        n_points=n_points, simulated_offsets=simulate_offsets, recovery_error=recovery,
        reference_id=ref.sensor_id,
    )
    return SyncResult(times=grid, aligned=aligned, present=present, metrics=metrics)


__all__ = [
    'SensorSeries', 'SyncMetrics', 'SyncResult',
    'estimate_clock_offset', 'synchronize',
]
