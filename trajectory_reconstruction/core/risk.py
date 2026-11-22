"""
智能风险评估 — 多因子可解释评分 + 分级处置建议

对应申报书创新点"基于自我决策的智能风险评估": 把重建后的三维轨迹、KinFlow 的
未来预测、传感器证据质量统一到一套可解释的评分体系里，输出风险等级、成因拆解、
时间线、关键事件与处置建议，而不是黑箱地给一个数字。

因子（默认权重）:
  1. 空间接近 proximity  0.30   —— 目标到防护区边界的距离
  2. 接近态势 closing    0.25   —— 逼近速度与预计到达时间 TTC
  3. 高度威胁 altitude   0.10   —— 低空目标更易实施抵近侦察/投放
  4. 机动强度 maneuver   0.10   —— 速度、加速度、曲率（规避/突防特征）
  5. 预测入侵 prediction 0.20   —— 未来视界内预测轨迹是否进入防护区
  6. 滞留时长 dwell      0.05   —— 已在防护区内的持续时间

总分 = 100 × Σ(w_i·s_i)/Σw_i，分级阈值默认 25/50/75。
证据置信度（传感器数量、融合不确定度）单独报告，不与威胁评分混为一谈。

纯 numpy / 标准库实现。
"""

from dataclasses import dataclass, field
import math

import numpy as np


@dataclass
class ProtectedZone:
    """防护区（圆柱近似）: 水平圆 + 高度带。"""

    center: tuple = (6.0, 2.0, 4.0)
    radius: float = 5.0
    y_min: float = 0.0
    y_max: float = 30.0

    @property
    def cx(self) -> float:
        return float(self.center[0])

    @property
    def cz(self) -> float:
        return float(self.center[2])

    def distance_h(self, p: np.ndarray) -> float:
        """水平面内到中心的距离。"""
        return float(math.hypot(p[0] - self.cx, p[2] - self.cz))

    def inside(self, p: np.ndarray) -> bool:
        return self.distance_h(p) <= self.radius and self.y_min <= p[1] <= self.y_max

    def to_dict(self) -> dict:
        return {
            'center': [round(float(c), 3) for c in self.center],
            'radius': float(self.radius),
            'y_min': float(self.y_min), 'y_max': float(self.y_max),
        }


@dataclass
class RiskWeights:
    proximity: float = 0.30
    closing: float = 0.25
    altitude: float = 0.10
    maneuver: float = 0.10
    prediction: float = 0.20
    dwell: float = 0.05

    def as_dict(self) -> dict:
        return {
            'proximity': self.proximity, 'closing': self.closing,
            'altitude': self.altitude, 'maneuver': self.maneuver,
            'prediction': self.prediction, 'dwell': self.dwell,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'RiskWeights':
        base = cls()
        if not data:
            return base
        for key in base.as_dict():
            if key in data and data[key] is not None:
                setattr(base, key, float(data[key]))
        return base


# 等级: (下限, 名称, 颜色, 处置要点)
LEVELS = [
    (75.0, '紧急', '#ff4d4f', '立即启动反制预案并同步告警'),
    (50.0, '高', '#ff9f43', '提升跟踪等级，预置拦截方案'),
    (25.0, '关注', '#f6c445', '持续监视并完整记录证据'),
    (0.0, '低', '#52c41a', '保持常规监视'),
]


def _clip01(x: float) -> float:
    return float(min(max(x, 0.0), 1.0))


def _level_of(score: float, thresholds: list = None) -> tuple:
    thresholds = thresholds or [75.0, 50.0, 25.0]
    for i, level in enumerate(LEVELS):
        if score >= thresholds[min(i, len(thresholds) - 1)]:
            return level
    return LEVELS[-1]


def _distance_series(zone: ProtectedZone, points: np.ndarray) -> np.ndarray:
    return np.sqrt((points[:, 0] - zone.cx) ** 2 + (points[:, 2] - zone.cz) ** 2)


def _slope(xs: np.ndarray, ys: np.ndarray) -> float:
    """最小二乘斜率（对 NaN 鲁棒）。"""
    mask = np.isfinite(xs) & np.isfinite(ys)
    if mask.sum() < 2:
        return 0.0
    x, y = xs[mask], ys[mask]
    if np.ptp(x) < 1e-9:
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def _prediction_factor(zone: ProtectedZone, prediction: dict,
                       horizon: float, last_time: float) -> tuple[float, dict]:
    """未来预测因子: 是否/多久进入防护区。"""
    if not prediction or not prediction.get('positions'):
        return 0.0, {'inside': False, 'eta': None, 'min_distance': None,
                     'detail': '无预测输入'}
    pos = np.asarray(prediction['positions'], dtype=float)
    ts = np.asarray(prediction.get('times') or [], dtype=float)
    finite = np.isfinite(pos).all(axis=1)
    if not finite.any():
        return 0.0, {'inside': False, 'eta': None, 'min_distance': None,
                     'detail': '预测数据无效'}
    pos = pos[finite]
    if len(ts) == len(finite):
        ts = ts[finite]

    dists = _distance_series(zone, pos)
    inside_mask = (dists <= zone.radius) & (pos[:, 1] >= zone.y_min) & (pos[:, 1] <= zone.y_max)
    min_dist = float(np.min(dists - zone.radius))

    if inside_mask.any():
        k = int(np.argmax(inside_mask))
        eta = float(ts[k] - last_time) if len(ts) else None
        return 1.0, {'inside': True, 'eta': round(eta, 3) if eta is not None else None,
                     'min_distance': round(min_dist, 3),
                     'detail': f'预测 {"%.1f" % eta if eta is not None else "?"} s 后进入防护区'}

    score = _clip01(1.0 - max(min_dist, 0.0) / 15.0) * 0.6
    return score, {'inside': False, 'eta': None, 'min_distance': round(min_dist, 3),
                   'detail': f'预测轨迹最近仍距边界 {max(min_dist, 0.0):.1f} m'}


def _compute_factors(times: np.ndarray, points: np.ndarray, zone: ProtectedZone,
                     weights: RiskWeights, prediction: dict, horizon: float,
                     approach_range: float = 20.0) -> tuple[dict, dict]:
    """计算各因子得分与统计量。"""
    last = points[-1]
    dists = _distance_series(zone, points)
    d_now = float(dists[-1])
    inside_mask = (dists <= zone.radius) & (points[:, 1] >= zone.y_min) & (points[:, 1] <= zone.y_max)
    inside_now = bool(inside_mask[-1])

    # 1) 空间接近
    boundary = d_now - zone.radius
    proximity = 1.0 if inside_now else _clip01(1.0 - max(boundary, 0.0) / approach_range)

    # 2) 接近态势
    k = min(len(times), 6)
    closing_speed = -_slope(times[-k:], dists[-k:])          # >0 表示在接近
    if inside_now:
        closing = 1.0
        ttc = 0.0
    elif closing_speed > 0.05:
        ttc = max(boundary, 0.0) / closing_speed
        closing = _clip01(1.0 - ttc / 30.0)
    else:
        ttc = None
        closing = 0.15 if closing_speed > -0.05 else 0.05

    # 3) 高度威胁
    y = float(last[1])
    if y < zone.y_min:
        altitude = 1.0
    elif y > zone.y_max:
        altitude = 0.4
    else:
        altitude = _clip01(1.0 - (y - 2.0) / 25.0)

    # 4) 机动强度
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.1
    disp = np.linalg.norm(np.diff(points, axis=0), axis=1)
    speed = disp / max(dt, 1e-6)
    v_max = float(np.nanmax(speed)) if len(speed) else 0.0
    if len(points) >= 3:
        v = np.gradient(points, axis=0) / max(dt, 1e-6)
        a = np.gradient(v, axis=0) / max(dt, 1e-6)
        a_max = float(np.nanmax(np.linalg.norm(a, axis=1)))
        d1 = points[1:-1] - points[:-2]
        d2 = points[2:] - points[1:-1]
        cross = np.linalg.norm(np.cross(d1, d2), axis=1)
        n1 = np.linalg.norm(d1, axis=1)
        curv = np.where(n1 > 1e-6, cross / np.maximum(n1 ** 3, 1e-9), 0.0)
        c_max = float(np.nanmax(curv)) if len(curv) else 0.0
    else:
        a_max, c_max = 0.0, 0.0
    maneuver = _clip01(0.5 * min(c_max / 0.05, 1.0)
                       + 0.3 * min(a_max / 8.0, 1.0)
                       + 0.2 * min(v_max / 15.0, 1.0))

    # 5) 预测入侵
    pred_score, pred_info = _prediction_factor(zone, prediction, horizon, float(times[-1]))

    # 6) 滞留时长
    if inside_mask.any():
        idx = np.where(inside_mask)[0]
        dwell_s = float(times[idx[-1]] - times[idx[0]]) + dt
    else:
        dwell_s = 0.0
    dwell = _clip01(dwell_s / 8.0)

    scores = {
        'proximity': proximity, 'closing': closing, 'altitude': altitude,
        'maneuver': maneuver, 'prediction': pred_score, 'dwell': dwell,
    }
    stats = {
        'last_position': [round(float(v), 4) for v in last],
        'distance_to_center': round(d_now, 3),
        'distance_to_boundary': round(boundary, 3),
        'inside': inside_now,
        'closing_speed': round(closing_speed, 3),
        'ttc': round(ttc, 2) if ttc is not None else None,
        'altitude': round(y, 3),
        'max_speed': round(v_max, 3),
        'max_accel': round(a_max, 3),
        'max_curvature': round(c_max, 5),
        'inside_duration': round(dwell_s, 2),
        'prediction': pred_info,
    }
    return scores, stats


def _recommendations(level: str, scores: dict, stats: dict, evidence: dict,
                     prediction: dict) -> list[dict]:
    """规则化处置建议（按优先级排序）。"""
    out = []
    if level == '紧急':
        out.append({'action': '立即启动反制预案', 'priority': 1,
                    'reason': '目标已进入或紧贴核心防护区'})
        out.append({'action': '同步安保与空管告警', 'priority': 1,
                    'reason': '高风险目标需人工确认与联动处置'})
    elif level == '高':
        out.append({'action': '提升跟踪等级并预置拦截', 'priority': 2,
                    'reason': '目标持续接近防护区，预计短时间内构成威胁'})

    pred = stats.get('prediction') or {}
    if pred.get('inside') and pred.get('eta') is not None:
        pos = prediction.get('positions') or []
        if pos:
            entry = [round(float(v), 2) for v in pos[0]]
            out.append({'action': f'在预计入侵点 {entry} 预置拦截', 'priority': 2,
                        'reason': f'预测 {pred["eta"]:.1f} s 后进入防护区'})
    if scores['closing'] >= 0.6 and not stats['inside']:
        out.append({'action': '提前预警并呼叫处置单元', 'priority': 2,
                    'reason': f'逼近速度 {stats["closing_speed"]:.2f} m/s，TTC '
                              f'{stats["ttc"] if stats["ttc"] is not None else "-"} s'})

    degraded = [k for k, v in (evidence.get('sensor_health') or {}).items()
                if v.get('status') in ('degraded', 'lost')]
    if degraded:
        out.append({'action': '调整传感器分工或补盲', 'priority': 3,
                    'reason': '以下传感器状态异常: ' + ', '.join(degraded)})
    if evidence.get('confidence', 1.0) < 0.5:
        out.append({'action': '增加观测或人工复核', 'priority': 3,
                    'reason': '当前证据置信度偏低，避免误报'})
    if stats['inside_duration'] > 2:
        out.append({'action': '完整记录取证并持续跟踪', 'priority': 3,
                    'reason': f'已在防护区内停留 {stats["inside_duration"]:.1f} s'})

    out.append({'action': '持续跟踪并滚动更新风险评估', 'priority': 4,
                'reason': '按主时间栅格周期性重算'})
    seen, uniq = set(), []
    for r in sorted(out, key=lambda x: x['priority']):
        if r['action'] in seen:
            continue
        seen.add(r['action'])
        uniq.append(r)
    return uniq[:6]


def _timeline(times: np.ndarray, points: np.ndarray, zone: ProtectedZone,
              weights: RiskWeights) -> list[float]:
    """逐时刻风险分（用前缀轨迹评估，体现风险演化）。"""
    scores = []
    for i in range(3, len(times) + 1):
        s, _ = _compute_factors(times[:i], points[:i], zone, weights, {}, 0.0)
        total = sum(weights.as_dict()[k] * s[k] for k in s)
        wsum = sum(weights.as_dict().values())
        scores.append(round(100.0 * total / max(wsum, 1e-9), 2))
    return [scores[0], scores[0]] + scores if len(scores) >= 1 else []


def assess(
    times,
    positions,
    zone: ProtectedZone = None,
    prediction: dict = None,
    uncertainty=None,
    sensor_health: dict = None,
    weights: RiskWeights = None,
    thresholds: list = None,
    horizon: float = 6.0,
) -> dict:
    """
    综合风险评估。

    Args:
        times:        主时间栅格 (M,)
        positions:    重建后的三维轨迹 (M,3)
        zone:         防护区
        prediction:   {"times": [...], "positions": [[x,y,z], ...]} 未来预测（可空）
        uncertainty:  逐点融合不确定度 (M,)（可空）
        sensor_health: 融合模块给出的传感器健康度（可空）
        weights:      因子权重
        thresholds:   分级阈值 [紧急, 高, 关注]
        horizon:      预测视界（秒），仅用于展示

    Returns:
        dict: score / level / factors / recommendations / timeline / events / summary …
    """
    zone = zone or ProtectedZone()
    weights = weights or RiskWeights()
    thresholds = thresholds or [75.0, 50.0, 25.0]

    t = np.asarray(times, dtype=float)
    p = np.asarray(positions, dtype=float)
    finite = np.isfinite(p).all(axis=1) & np.isfinite(t)
    if finite.sum() < 4:
        return {
            'score': 0.0, 'level': '低', 'color': LEVELS[-1][2],
            'factors': [], 'recommendations': [], 'timeline': {'times': [], 'scores': []},
            'events': {}, 'summary': '有效轨迹点不足，无法评估', 'zone': zone.to_dict(),
        }
    t, p = t[finite], p[finite]

    scores, stats = _compute_factors(t, p, zone, weights, prediction or {}, horizon)

    w = weights.as_dict()
    wsum = sum(w.values())
    total = 100.0 * sum(w[k] * scores[k] for k in scores) / max(wsum, 1e-9)
    total = round(float(np.clip(total, 0.0, 100.0)), 2)
    level_name, level_color, level_advice = _level_of(total, thresholds)[1:]

    labels = {
        'proximity': ('空间接近', lambda v: f'距边界 {stats["distance_to_boundary"]:.1f} m'),
        'closing': ('接近态势', lambda v: f'逼近 {stats["closing_speed"]:.2f} m/s'
                                            + (f'，TTC {stats["ttc"]:.1f} s' if stats['ttc'] is not None else '')),
        'altitude': ('高度威胁', lambda v: f'高度 {stats["altitude"]:.1f} m'),
        'maneuver': ('机动强度', lambda v: f'最大速度 {stats["max_speed"]:.1f} m/s，'
                                           f'最大加速度 {stats["max_accel"]:.1f} m/s²'),
        'prediction': ('预测入侵', lambda v: stats['prediction'].get('detail', '')),
        'dwell': ('滞留时长', lambda v: f'区内停留 {stats["inside_duration"]:.1f} s'),
    }
    factors = []
    for key, value in scores.items():
        label, detail_fn = labels[key]
        factors.append({
            'key': key, 'label': label,
            'score': round(float(value), 4),
            'weight': float(w[key]),
            'contribution': round(100.0 * w[key] * value / max(wsum, 1e-9), 2),
            'detail': detail_fn(value),
        })

    # 证据置信度（与威胁评分分开报告）
    conf = 1.0
    if uncertainty is not None:
        u = np.asarray(uncertainty, dtype=float)
        u = u[np.isfinite(u)]
        if len(u):
            conf = float(np.exp(-np.mean(u) / 5.0))
    if sensor_health:
        healths = [v.get('status') for v in sensor_health.values()]
        n_ok = sum(1 for s in healths if s == 'normal')
        conf *= min(1.0, 0.5 + 0.25 * n_ok) if healths else 1.0
    conf = round(float(np.clip(conf, 0.05, 1.0)), 4)

    evidence = {
        'confidence': conf,
        'sensor_health': sensor_health or {},
        'mean_uncertainty': round(float(np.nanmean(np.asarray(uncertainty, dtype=float))), 3)
        if uncertainty is not None and np.isfinite(np.asarray(uncertainty, dtype=float)).any() else None,
        'n_frames': int(len(t)),
        'hz': round(float(1.0 / max(np.median(np.diff(t)), 1e-6)), 2) if len(t) > 1 else 0.0,
    }

    # 关键事件
    dists = _distance_series(zone, p)
    inside = (dists <= zone.radius) & (p[:, 1] >= zone.y_min) & (p[:, 1] <= zone.y_max)
    tl = _timeline(t, p, zone, weights)
    peak_idx = int(np.argmax(tl)) if tl else 0
    first_in = int(np.argmax(inside)) if inside.any() else None
    events = {
        'zone_entry': {'time': round(float(t[first_in]), 3), 'position': [round(float(v), 2) for v in p[first_in]]}
        if first_in is not None else None,
        'peaked': {'time': round(float(t[peak_idx]), 3) if tl else None,
                   'score': round(float(tl[peak_idx]), 2) if tl else None},
        'predicted_entry': stats['prediction'].get('eta'),
        'n_inside_frames': int(inside.sum()),
    }

    recommendations = _recommendations(level_name, scores, stats, evidence, prediction or {})

    dominant = max(factors, key=lambda f: f['contribution']) if factors else None
    summary = (f"风险等级：{level_name}（{total:.0f}/100，证据置信度 {conf:.2f}）。"
               + (f"主要成因：{dominant['label']}（{dominant['detail']}）。" if dominant else '')
               + (f" 首要处置：{recommendations[0]['action']}。" if recommendations else ''))

    return {
        'score': total,
        'level': level_name,
        'color': level_color,
        'level_advice': level_advice,
        'factors': factors,
        'confidence': conf,
        'evidence': evidence,
        'stats': stats,
        'zone': zone.to_dict(),
        'events': events,
        'recommendations': recommendations,
        'timeline': {'times': [round(float(x), 3) for x in t], 'scores': tl[:len(t)]},
        'summary': summary,
        'weights': w,
    }


__all__ = ['ProtectedZone', 'RiskWeights', 'LEVELS', 'assess']
