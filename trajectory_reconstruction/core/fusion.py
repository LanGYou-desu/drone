"""
多传感器协同融合 — 数据关联 + 逆方差加权 + 传感器健康度

时间同步只解决"对齐"，真正的协同还需要判断各传感器在同一时刻观测到的是否为
同一个目标，并按测量精度融合成一条更可信的轨迹:

  1. 关联: 以精度最高的传感器为种子，用空间门限 gate_m 聚类，剔除被遮挡/误检/串扰的点；
  2. 融合: 逆方差加权均值 p = Σw_i p_i / Σw_i，w_i = 1/σ_i²，
     融合不确定度 σ = 1/√(Σw_i)；单传感器时按 single_sensor_inflation 放大不确定度；
  3. 创新量: 每个传感器与融合结果的偏差，用于识别异常传感器；
  4. 健康度: 覆盖率、被剔除比例、平均创新量 -> normal / degraded / lost。

纯 numpy 实现，无外部依赖。
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FusionConfig:
    """融合参数。"""

    gate_m: float = 3.0                       # 关联门限（米）
    sigmas: dict[str, float] = field(default_factory=dict)   # 各传感器测量噪声（米）
    default_sigma: float = 1.5
    single_sensor_inflation: float = 1.5      # 单传感器时的不确定度放大倍数
    min_sensors_for_fusion: int = 2
    continuity: bool = True                   # 按航迹连续性选择关联种子（多目标/误检更稳）
    max_speed: float = 40.0                   # 关联预测的速度上限（m/s）

    def sigma_of(self, sensor_id: str) -> float:
        return float(self.sigmas.get(sensor_id, self.default_sigma))


@dataclass
class FusionResult:
    """融合结果（逐主时间栅格）。"""

    times: np.ndarray                  # (M,)
    positions: np.ndarray              # (M,3) 无观测处为 NaN
    sigmas: np.ndarray                 # (M,)  融合不确定度（米）
    n_sensors: np.ndarray              # (M,)  参与融合的传感器数量
    used: list[list[str]]              # 每个时刻参与融合的传感器
    rejected: list[list[str]]          # 每个时刻被关联门限剔除的传感器
    innovations: dict[str, np.ndarray] # 各传感器相对融合结果的偏差（米）
    health: dict[str, dict]            # 传感器健康度
    agreement: float                   # 至少 2 个传感器一致的时刻占比
    selected: list = None              # 参与融合的一致传感器子集
    excluded: list = None              # 因与其他传感器长期不一致而排除的传感器
    track_jumps: int = 0               # 航迹重捕获次数（融合点跳变 > 2×门限）
    to_dict_samples: int = 0

    def to_dict(self) -> dict:
        return {
            'times': [round(float(t), 4) for t in self.times],
            'positions': [[None, None, None] if not np.isfinite(p).all()
                          else [round(float(v), 4) for v in p] for p in self.positions],
            'sigmas': [round(float(s), 4) if np.isfinite(s) else None for s in self.sigmas],
            'n_sensors': [int(n) for n in self.n_sensors],
            'used': self.used,
            'rejected': self.rejected,
            'innovations': {
                sid: [round(float(v), 4) if np.isfinite(v) else None for v in arr]
                for sid, arr in self.innovations.items()
            },
            'health': self.health,
            'agreement': round(self.agreement, 4),
            'selected': list(self.selected or []),
            'excluded': list(self.excluded or []),
            'track_jumps': int(self.track_jumps),
        }


def _status(coverage: float, rejection: float, innovation: float, gate: float) -> str:
    if coverage < 0.2:
        return 'lost'
    if coverage < 0.5 or rejection > 0.3 or (np.isfinite(innovation) and innovation > gate):
        return 'degraded'
    return 'normal'


def _consistent_subset(times, aligned, present, sig_of, gate) -> list:
    """
    在融合前先选出"互相一致"的传感器子集。

    多目标/标定偏差场景下，若某传感器与其它传感器的中位距离长期超过门限，
    说明它观测的不是同一个目标；把它排除后，融合结果不会在目标之间来回跳。
    用并查集求一致连通分量，取规模最大者（同规模取噪声更小者）。
    """
    ids = list(aligned)
    n = len(ids)
    if n <= 1:
        return ids

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            common = present[ids[i]] & present[ids[j]]
            if common.sum() < 3:
                continue
            diff = np.linalg.norm(aligned[ids[i]][common] - aligned[ids[j]][common], axis=1)
            if float(np.median(diff)) <= gate:
                union(i, j)

    groups: dict[int, list] = {}
    for i, sid in enumerate(ids):
        groups.setdefault(find(i), []).append(sid)

    def rank(group):
        return (len(group), -float(np.mean([sig_of[s] for s in group])))

    return max(groups.values(), key=rank)


def fuse(
    times: np.ndarray,
    aligned: dict[str, np.ndarray],
    present: dict[str, np.ndarray],
    meta: dict[str, dict] = None,
    config: FusionConfig = None,
) -> FusionResult:
    """
    逐时刻关联并融合多传感器观测。

    Args:
        times:   主时间栅格 (M,)
        aligned: 各传感器对齐后的位置 (M,3)，缺测为 NaN
        present: 各传感器逐时刻是否有效 (M,)
        meta:    传感器元信息 {sid: {name, color, sigma?}}
        config:  融合参数

    Returns:
        FusionResult
    """
    cfg = config or FusionConfig()
    meta = meta or {}
    ids = [sid for sid in aligned if sid in present]
    if not ids:
        raise ValueError('没有可融合的传感器数据')

    m = len(times)
    positions = np.full((m, 3), np.nan)
    sigmas = np.full(m, np.nan)
    n_sensors = np.zeros(m, dtype=int)
    used: list[list[str]] = [[] for _ in range(m)]
    rejected: list[list[str]] = [[] for _ in range(m)]
    innovations = {sid: np.full(m, np.nan) for sid in ids}

    sig_of = {sid: float(meta.get(sid, {}).get('sigma', cfg.sigma_of(sid))) for sid in ids}
    selected = _consistent_subset(times, aligned, present, sig_of, cfg.gate_m)
    excluded = [sid for sid in ids if sid not in selected]
    order = sorted(selected, key=lambda s: sig_of[s])
    dt = float(np.median(np.diff(times))) if m > 1 else 0.1

    last_pos = None
    last_vel = np.zeros(3)
    track_jumps = 0

    for i in range(m):
        avail = [sid for sid in order if present[sid][i] and np.isfinite(aligned[sid][i]).all()]
        if not avail:
            continue

        # 关联种子: 有历史航迹时优先选最接近预测位置的传感器（保持航迹连续、抑制误检），
        # 否则退化为"精度最高者"。
        predicted = last_pos + last_vel * dt if (cfg.continuity and last_pos is not None) else None
        if predicted is not None:
            def _dist(sid):
                return float(np.linalg.norm(aligned[sid][i] - predicted))
            seed = min(avail, key=_dist)
            if _dist(seed) > cfg.gate_m * 2.0:     # 预测处无观测 -> 重新捕获
                seed = avail[0]
        else:
            seed = avail[0]

        seed_pos = aligned[seed][i]
        chosen = [seed]
        for sid in avail:
            if sid != seed and np.linalg.norm(aligned[sid][i] - seed_pos) <= cfg.gate_m:
                chosen.append(sid)

        weights = np.array([1.0 / max(sig_of[sid], 1e-3) ** 2 for sid in chosen])
        pts = np.stack([aligned[sid][i] for sid in chosen], axis=0)
        fused = (weights[:, None] * pts).sum(axis=0) / weights.sum()
        sig = float(1.0 / np.sqrt(weights.sum()))
        if len(chosen) == 1:
            sig *= cfg.single_sensor_inflation
        if len(chosen) < cfg.min_sensors_for_fusion:
            sig = max(sig, cfg.gate_m / 3.0)   # 证据不足时保守放大不确定度

        # 航迹连续性状态
        if last_pos is not None:
            jump = float(np.linalg.norm(fused - last_pos))
            if jump > cfg.gate_m * 2.0:
                track_jumps += 1
            vel = (fused - last_pos) / max(dt, 1e-6)
            speed = float(np.linalg.norm(vel))
            if speed > cfg.max_speed:
                vel = vel * (cfg.max_speed / speed)
            last_vel = 0.6 * last_vel + 0.4 * vel
        last_pos = fused

        positions[i] = fused
        sigmas[i] = sig
        n_sensors[i] = len(chosen)
        used[i] = chosen
        rejected[i] = [sid for sid in avail if sid not in chosen]
        for sid in avail:
            innovations[sid][i] = float(np.linalg.norm(aligned[sid][i] - fused))

    health = {}
    for sid in ids:
        cov = float(present[sid].mean()) if m else 0.0
        sel = present[sid]
        rej_rate = float(np.mean([sid in rejected[i] for i in range(m) if sel[i]])) if sel.any() else 0.0
        innov = innovations[sid]
        mean_innov = float(np.nanmean(innov)) if np.isfinite(innov).any() else float('nan')
        info = meta.get(sid, {})
        health[sid] = {
            'name': info.get('name', sid),
            'color': info.get('color', '#999999'),
            'sigma': round(sig_of[sid], 3),
            'coverage': round(cov, 4),
            'rejection_rate': round(rej_rate, 4),
            'mean_innovation': round(mean_innov, 4) if np.isfinite(mean_innov) else None,
            'status': _status(cov, rej_rate, mean_innov, cfg.gate_m),
            'points': int(np.sum(sel)),
        }

    multi = np.array([len(used[i]) >= 2 for i in range(m)])
    agreement = float(multi.mean()) if m else 0.0

    return FusionResult(
        times=np.asarray(times, dtype=np.float64),
        positions=positions, sigmas=sigmas, n_sensors=n_sensors,
        used=used, rejected=rejected, innovations=innovations,
        health=health, agreement=agreement, selected=selected, excluded=excluded,
        track_jumps=track_jumps,
    )


__all__ = ['FusionConfig', 'FusionResult', 'fuse']
