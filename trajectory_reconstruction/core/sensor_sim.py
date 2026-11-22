"""
多传感器协同观测仿真 — 同一目标的异构传感器观测生成

实际外场试验前，需要可重复地在软件内验证"时间同步 → 关联融合 → 三维重建"
整条链路是否真的带来增益。本模块以一条已知真值轨迹为输入，模拟 N 个传感器
对**同一目标**的观测，注入真实系统中会遇到的各种缺陷:

  · 各传感器采样率不同（如可见光 5 Hz、红外 4 Hz、雷达 10 Hz）
  · 时钟偏置与微小漂移（offset_s / drift）
  · 测量噪声（sigma 米）
  · 遮挡盲区（occlusion windows，目标被建筑/云台视野遮挡）
  · 随机丢帧（dropout）
  · 偶发误检（outlier，位置跳变数米）

输出各传感器观测序列 + 真值，供同步/融合/重建模块评估精度增益。
纯 numpy 实现，固定随机种子可复现。
"""

from dataclasses import dataclass, field

import numpy as np


# 默认传感器配置（贴近本项目演示参数：可见光/红外/雷达）
DEFAULT_SPECS = [
    # 注意: 参考传感器取采样点最多者（此处为雷达），其偏置设为 0 便于对照真值时间
    {'id': 'visible', 'name': '可见光', 'color': '#ff6b6b', 'rate': 5.0,
     'sigma': 0.35, 'offset': 0.10, 'drift': 0.0, 'occlusion': [[3.2, 4.6]],
     'dropout': 0.03, 'outlier': 0.0, 'bias': [0.0, 0.0, 0.0]},
    {'id': 'infrared', 'name': '红外', 'color': '#4ecdc4', 'rate': 4.0,
     'sigma': 0.60, 'offset': 0.20, 'drift': 0.0, 'occlusion': [[7.0, 8.2]],
     'dropout': 0.05, 'outlier': 0.02, 'bias': [0.15, -0.1, 0.0]},
    {'id': 'radar', 'name': '雷达', 'color': '#ffe66d', 'rate': 10.0,
     'sigma': 0.25, 'offset': 0.00, 'drift': 0.0, 'occlusion': [],
     'dropout': 0.02, 'outlier': 0.01, 'bias': [-0.1, 0.05, 0.1]},
]


@dataclass
class SimSensor:
    """单个仿真传感器的观测。"""

    id: str
    name: str
    color: str
    sigma: float
    rate: float
    offset: float
    times: np.ndarray
    positions: np.ndarray

    def to_dict(self) -> dict:
        return {
            'id': self.id, 'name': self.name, 'color': self.color,
            'sigma': self.sigma, 'rate': self.rate, 'offset': self.offset,
            'times': [round(float(t), 4) for t in self.times],
            'positions': [[round(float(v), 4) for v in p] for p in self.positions],
        }


@dataclass
class SimResult:
    """仿真数据集: 真值 + 各传感器观测 + 注入参数。"""

    truth_times: np.ndarray
    truth_positions: np.ndarray
    sensors: list[SimSensor]
    injected: dict = field(default_factory=dict)
    duration: float = 0.0

    def to_dict(self) -> dict:
        return {
            'duration': round(self.duration, 3),
            'truth': {
                'times': [round(float(t), 4) for t in self.truth_times],
                'positions': [[round(float(v), 4) for v in p] for p in self.truth_positions],
            },
            'sensors': [s.to_dict() for s in self.sensors],
            'injected': self.injected,
        }


def _smooth_track(points: np.ndarray, window: int = 9) -> np.ndarray:
    """滑动平均平滑（端点保持）。"""
    n = len(points)
    if n < 3 or window <= 1:
        return points.copy()
    half = window // 2
    out = points.copy()
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out[i] = points[lo:hi].mean(axis=0)
    return out


def build_truth(times, positions, hz: float = 20.0,
                extend: float = 0.0, smooth_window: int = 9) -> tuple[np.ndarray, np.ndarray]:
    """
    把一条离散轨迹提升为高频平滑真值。

    Args:
        times:     原始时间戳 (N,)
        positions: 原始位置 (N,3)
        hz:        真值频率
        extend:    末尾外推时长（秒），用末速度线性延伸
        smooth_window: 平滑窗口（点数）
    """
    t = np.asarray(times, dtype=float)
    p = np.asarray(positions, dtype=float)
    order = np.argsort(t)
    t, p = t[order], p[order]
    keep = np.concatenate([[True], np.diff(t) > 1e-9]) if len(t) else np.zeros(0, bool)
    t, p = t[keep], p[keep]
    if len(t) < 2:
        raise ValueError('真值轨迹至少需要 2 个点')

    t_end = t[-1] + max(extend, 0.0)
    grid = np.arange(t[0], t_end, 1.0 / hz)
    pos = np.stack([np.interp(grid, t, p[:, axis]) for axis in range(3)], axis=1)

    if extend > 0 and len(t) >= 2:
        v = (p[-1] - p[-2]) / max(t[-1] - t[-2], 1e-6)
        tail = grid > t[-1]
        if tail.any():
            pos[tail] = p[-1] + np.outer(grid[tail] - t[-1], v)

    pos = _smooth_track(pos, smooth_window)
    return grid, pos


def simulate_multi_sensor(
    truth_times: np.ndarray,
    truth_positions: np.ndarray,
    specs: list[dict] = None,
    seed: int = 42,
) -> SimResult:
    """
    以真值轨迹生成多传感器观测。

    Args:
        truth_times:    真值时间戳（秒）
        truth_positions:(N,3) 真值位置
        specs:          传感器配置（缺省 DEFAULT_SPECS）
        seed:           随机种子

    Returns:
        SimResult
    """
    specs = specs or DEFAULT_SPECS
    rng = np.random.default_rng(seed)
    tt = np.asarray(truth_times, dtype=float)
    tp = np.asarray(truth_positions, dtype=float)
    t0, t1 = float(tt[0]), float(tt[-1])
    duration = t1 - t0

    sensors: list[SimSensor] = []
    injected = {'offsets': {}, 'rates': {}, 'sigmas': {}, 'occlusion': {}}

    for spec in specs:
        sid = spec['id']
        rate = float(spec.get('rate', 5.0))
        sigma = float(spec.get('sigma', 0.5))
        offset = float(spec.get('offset', 0.0))
        drift = float(spec.get('drift', 0.0))
        dropout = float(spec.get('dropout', 0.0))
        outlier_p = float(spec.get('outlier', 0.0))
        bias = np.asarray(spec.get('bias', [0.0, 0.0, 0.0]), dtype=float)
        occlusion = spec.get('occlusion', []) or []

        # 传感器自己的时间轴 = 真值时刻 + 时钟偏置；按自身采样率抽取
        n = max(int(duration * rate) + 1, 4)
        t_local = np.linspace(0.0, duration, n)
        t_obs = t0 + t_local + offset + drift * t_local + rng.uniform(-0.04, 0.04, n) / rate
        pos = np.stack([np.interp(t_obs - offset, tt, tp[:, axis]) for axis in range(3)], axis=1)

        # 系统偏差（安装/标定残差）+ 白噪声
        pos = pos + bias + rng.normal(0.0, sigma, size=pos.shape)

        # 遮挡盲区（按真值时刻判定）
        keep = np.ones(n, dtype=bool)
        truth_t = t_obs - offset
        for window in occlusion:
            lo, hi = float(window[0]), float(window[1])
            keep &= ~((truth_t >= lo) & (truth_t <= hi))

        # 随机丢帧
        if dropout > 0:
            keep &= rng.random(n) > dropout

        # 偶发误检: 位置跳变数米
        if outlier_p > 0:
            bad = rng.random(n) < outlier_p
            if bad.any():
                pos[bad] += rng.normal(0.0, 4.0, size=(int(bad.sum()), 3))

        t_obs, pos = t_obs[keep], pos[keep]
        order = np.argsort(t_obs)
        sensors.append(SimSensor(
            id=sid, name=spec.get('name', sid), color=spec.get('color', '#999999'),
            sigma=sigma, rate=rate, offset=offset,
            times=t_obs[order], positions=pos[order],
        ))
        injected['offsets'][sid] = offset
        injected['rates'][sid] = rate
        injected['sigmas'][sid] = sigma
        injected['occlusion'][sid] = occlusion

    injected['seed'] = seed
    return SimResult(truth_times=tt, truth_positions=tp, sensors=sensors,
                     injected=injected, duration=duration)


def default_scenario_specs(**overrides) -> list[dict]:
    """生成默认传感器配置的可变副本（UI 参数覆盖）。"""
    import copy
    specs = copy.deepcopy(DEFAULT_SPECS)
    for sid, data in (overrides.get('sensors') or {}).items():
        for spec in specs:
            if spec['id'] == sid:
                spec.update(data)
    return specs


__all__ = [
    'DEFAULT_SPECS', 'SimSensor', 'SimResult',
    'build_truth', 'simulate_multi_sensor', 'default_scenario_specs',
]
