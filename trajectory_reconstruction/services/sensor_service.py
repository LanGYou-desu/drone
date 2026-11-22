"""
多传感器协同服务 — 数据采集 → 时间同步 → 关联融合 → 三维重建 → 智能评估

把 core/timesync、core/fusion、core/reconstruction、core/risk、core/sensor_sim
串成一条可复用的处理链路，供 REST API 与前端页面调用。

两种数据源:
  · live      —— 运行时各平台（可见光/红外/雷达/自选）的真实轨迹
  · simulated —— 以某条轨迹为真值，模拟多传感器对同一目标的异构观测
                 （不同采样率/噪声/时钟偏置/遮挡/丢帧/误检），用于定量验证协同增益
"""

import json

import numpy as np

from trajectory_reconstruction.core import sensor_sim
from trajectory_reconstruction.core.config.config_manager import ensure_config, save_config
from trajectory_reconstruction.core.fusion import FusionConfig, fuse
from trajectory_reconstruction.core.reconstruction import ReconstructionConfig, reconstruct
from trajectory_reconstruction.core.risk import ProtectedZone, RiskWeights, assess
from trajectory_reconstruction.core.state import METHOD_ORDER, detection_methods
from trajectory_reconstruction.core.timesync import SensorSeries, synchronize

# 处理链路结果缓存：页面切换/参数不变时避免重复计算（同步→融合→重建约 60~130 ms）
_PIPELINE_CACHE: dict = {}
_ASSESS_CACHE: dict = {}
_CACHE_MAX = 8


def _data_signature(series) -> tuple:
    """用各传感器的点数与时间范围作为数据版本指纹。"""
    return tuple(
        (s.sensor_id, len(s.times), round(float(s.times[0]), 3), round(float(s.times[-1]), 3))
        for s in series
    )


def _cache_put(cache: dict, key: tuple, value: dict):
    if len(cache) >= _CACHE_MAX:
        cache.pop(next(iter(cache)))
    cache[key] = value


_DEFAULTS = {
    'hz': 10.0,
    'max_offset_s': 0.6,
    'max_sample_gap_s': 0.5,
    'gate_m': 3.0,
    'default_sigma': 1.5,
    'sigmas': {'visible': 1.0, 'infrared': 1.5, 'radar': 0.6, 'self': 2.0},
    'max_gap_s': 0.8,
    'smooth_window': 5,
    'outlier_sigma': 3.5,
    'zone': {'center': [8.0, 12.0, 4.0], 'radius': 6.0, 'y_min': 5.0, 'y_max': 25.0},
    'thresholds': [75.0, 50.0, 25.0],
    'weights': {'proximity': 0.30, 'closing': 0.25, 'altitude': 0.10,
                'maneuver': 0.10, 'prediction': 0.20, 'dwell': 0.05},
    'prediction_horizon_s': 2.0,
    'prediction_points': 10,
}


# ── 配置 ──────────────────────────────────────────────

def get_collab_config() -> dict:
    """读取 config.json 中的协同配置（缺失字段回落默认值）。"""
    cfg = ensure_config()
    sync_cfg = cfg.get('synchronization', {}) or {}
    fusion_cfg = cfg.get('sensor_fusion', {}) or {}
    recon_cfg = cfg.get('reconstruction', {}) or {}
    assess_cfg = cfg.get('assessment', {}) or {}

    result = dict(_DEFAULTS)
    result['sigmas'] = dict(_DEFAULTS['sigmas'], **(fusion_cfg.get('sigmas') or {}))

    for key in ('hz', 'max_offset_s', 'max_sample_gap_s'):
        if key in sync_cfg:
            result[key] = sync_cfg[key]
    for key in ('gate_m', 'default_sigma'):
        if key in fusion_cfg:
            result[key] = fusion_cfg[key]
    for key in ('max_gap_s', 'smooth_window', 'outlier_sigma'):
        if key in recon_cfg:
            result[key] = recon_cfg[key]
    for key in ('zone', 'thresholds', 'weights', 'prediction_horizon_s', 'prediction_points'):
        if key in assess_cfg:
            result[key] = assess_cfg[key]
    return result


def update_collab_params(payload: dict) -> dict:
    """
    把界面修改的协同/评估参数写回 config.json。

    重建参数（同步频率/关联门限/平滑窗口/最大补洞）与防护区参数原本每次页面
    刷新都会回到默认值；持久化后刷新、重启都保持上次的设置，重建结果与界面一致。
    """
    cfg = ensure_config()
    changed = {}

    def _section(name: str) -> dict:
        return cfg.setdefault(name, {}) or {}

    if payload.get('hz') is not None:
        cfg['synchronization'] = _section('synchronization')
        cfg['synchronization']['hz'] = round(_clamp(payload['hz'], 2.0, 30.0, 10.0), 2)
        changed['hz'] = cfg['synchronization']['hz']
    if payload.get('gate_m') is not None:
        cfg['sensor_fusion'] = _section('sensor_fusion')
        cfg['sensor_fusion']['gate_m'] = round(_clamp(payload['gate_m'], 0.2, 20.0, 3.0), 2)
        changed['gate_m'] = cfg['sensor_fusion']['gate_m']
    if payload.get('smooth_window') is not None:
        cfg['reconstruction'] = _section('reconstruction')
        cfg['reconstruction']['smooth_window'] = int(_clamp(payload['smooth_window'], 1, 15, 5))
        changed['smooth_window'] = cfg['reconstruction']['smooth_window']
    if payload.get('max_gap_s') is not None:
        cfg['reconstruction'] = _section('reconstruction')
        cfg['reconstruction']['max_gap_s'] = round(_clamp(payload['max_gap_s'], 0.0, 5.0, 0.8), 2)
        changed['max_gap_s'] = cfg['reconstruction']['max_gap_s']
    if payload.get('prediction_horizon_s') is not None:
        cfg['assessment'] = _section('assessment')
        cfg['assessment']['prediction_horizon_s'] = round(
            _clamp(payload['prediction_horizon_s'], 0.2, 10.0, 2.0), 2)
        changed['prediction_horizon_s'] = cfg['assessment']['prediction_horizon_s']

    zone = payload.get('zone')
    if isinstance(zone, dict) and isinstance(zone.get('center'), (list, tuple)) and len(zone['center']) == 3:
        try:
            center = [round(float(v), 2) for v in zone['center']]
            radius = round(_clamp(zone.get('radius', 4.0), 0.5, 100.0, 4.0), 2)
            y_min = round(_clamp(zone.get('y_min', 0.0), -1000.0, 1000.0, 0.0), 2)
            y_max = round(_clamp(zone.get('y_max', 20.0), y_min + 0.5, 1000.0, 20.0), 2)
        except (TypeError, ValueError):
            center = radius = y_min = y_max = None
        if center is not None:
            cfg['assessment'] = _section('assessment')
            cfg['assessment']['zone'] = {'center': center, 'radius': radius,
                                         'y_min': y_min, 'y_max': y_max}
            changed['zone'] = cfg['assessment']['zone']

    if changed:
        save_config(cfg)
    return changed


def _clamp(value, lo, hi, default):
    try:
        return float(min(max(float(value), lo), hi))
    except (TypeError, ValueError):
        return float(default)


# ── 数据采集 ──────────────────────────────────────────

def _sigma_of(sensor_id: str, config: dict) -> float:
    return float((config.get('sigmas') or {}).get(sensor_id, config.get('default_sigma', 1.5)))


def collect_live_series(config: dict) -> list[SensorSeries]:
    """采集运行时各平台轨迹（排除综合合成轨迹）。"""
    series = []
    for mid in METHOD_ORDER:
        if mid == 'synthetic':
            continue
        data = detection_methods.get(mid)
        if not data:
            continue
        points = data.get('points') or []
        times = data.get('timestamps') or []
        if len(points) < 3 or len(times) != len(points):
            continue
        series.append(SensorSeries(
            mid, np.asarray(times, dtype=float), np.asarray(points, dtype=float),
            data.get('name', mid), data.get('color', '#999999'), _sigma_of(mid, config),
        ))
    return series


def _base_track() -> tuple[np.ndarray, np.ndarray]:
    """
    仿真真值基准。

    优先选**机动性最强**的真实平台轨迹（路径长度最大）：时钟偏置的可观测性
    取决于轨迹的机动性，匀速直线段几乎不携带时间信息；综合轨迹是各平台的加权
    平均，机动性往往被抹平，仅在无其他平台时使用。
    """
    candidates = []
    for mid in METHOD_ORDER:
        if mid == 'synthetic':
            continue
        data = detection_methods.get(mid)
        if not data:
            continue
        pts = data.get('points') or []
        ts = data.get('timestamps') or []
        if len(pts) < 5 or len(ts) != len(pts):
            continue
        arr = np.asarray(pts, dtype=float)
        path = float(np.linalg.norm(np.diff(arr, axis=0), axis=1).sum())
        candidates.append((path, mid, np.asarray(ts, dtype=float), arr))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        _, _, ts, pts = candidates[0]
        return ts, pts

    # 次选: 综合轨迹
    data = detection_methods.get('synthetic')
    if data and len(data.get('points') or []) >= 5:
        return (np.asarray(data['timestamps'], dtype=float),
                np.asarray(data['points'], dtype=float))

    # 兜底: 一条平滑的机动航线（12 s，半径 8 m 的螺旋 + 侧向摆动）
    t = np.arange(0.0, 12.0, 0.2)
    w = 2 * np.pi / 6.0
    p = np.stack([
        8 + 6 * np.cos(w * t),
        2.5 + 0.8 * np.sin(0.8 * w * t),
        4 + 6 * np.sin(w * t),
    ], axis=1)
    return t, p


def simulate_series(config: dict, payload: dict) -> tuple[list[SensorSeries], np.ndarray, np.ndarray, dict]:
    """生成同目标的多传感器观测（仿真场景）。"""
    sim_cfg = payload.get('sim') or {}
    seed = int(sim_cfg.get('seed', 42))
    truth_hz = _clamp(sim_cfg.get('truth_hz', 20.0), 5.0, 60.0, 20.0)
    extend = _clamp(sim_cfg.get('extend_s', 2.0), 0.0, 10.0, 2.0)

    base_t, base_p = _base_track()
    truth_t, truth_p = sensor_sim.build_truth(base_t, base_p, hz=truth_hz, extend=extend)

    specs = sensor_sim.default_scenario_specs(sensors=sim_cfg.get('sensors') or {})
    sim = sensor_sim.simulate_multi_sensor(truth_t, truth_p, specs=specs, seed=seed)

    series = [
        SensorSeries(s.id, s.times, s.positions, s.name, s.color, s.sigma)
        for s in sim.sensors
    ]
    return series, sim.truth_times, sim.truth_positions, sim.injected


# ── 处理链路 ──────────────────────────────────────────

def _accuracy(times, aligned, present, fused_pos, recon_pos, truth_t, truth_p) -> dict:
    """以真值为基准，比较单传感器 / 融合 / 重建的 RMSE。"""
    inside = (times >= truth_t[0]) & (times <= truth_t[-1])
    if not inside.any():
        return {}
    gt = np.stack([np.interp(times[inside], truth_t, truth_p[:, axis]) for axis in range(3)], axis=1)

    def _rmse(arr):
        a = arr[inside]
        ok = np.isfinite(a).all(axis=1)
        if not ok.any():
            return None
        return float(np.sqrt(np.mean(np.sum((a[ok] - gt[ok]) ** 2, axis=1))))

    per_sensor = {}
    for sid, arr in aligned.items():
        per_sensor[sid] = _rmse(arr)
    fused_rmse = _rmse(fused_pos)
    recon_rmse = _rmse(recon_pos)

    valid = {k: v for k, v in per_sensor.items() if v is not None}
    best_single = min(valid.values()) if valid else None
    result = {
        'per_sensor_rmse': {k: (round(v, 4) if v is not None else None) for k, v in per_sensor.items()},
        'fused_rmse': round(fused_rmse, 4) if fused_rmse is not None else None,
        'reconstructed_rmse': round(recon_rmse, 4) if recon_rmse is not None else None,
        'best_single_rmse': round(best_single, 4) if best_single is not None else None,
        'frames': int(inside.sum()),
    }
    if best_single and fused_rmse is not None:
        result['fusion_gain_pct'] = round(100.0 * (best_single - fused_rmse) / best_single, 2)
    if best_single and recon_rmse is not None:
        result['reconstruction_gain_pct'] = round(100.0 * (best_single - recon_rmse) / best_single, 2)
    return result


def run_pipeline(payload: dict = None) -> dict:
    """
    执行"同步 → 融合 → 重建"链路。

    Args:
        payload: {scenario: 'live'|'simulated', hz, gate_m, smooth_window, max_gap_s,
                  max_offset_s, sim: {...}, with_series: bool}

    Returns:
        dict（含同步指标、融合摘要、重建结果、仿真精度对比）
    """
    payload = payload or {}
    config = get_collab_config()
    scenario = payload.get('scenario', 'live')

    hz = _clamp(payload.get('hz', config['hz']), 2.0, 30.0, config['hz'])
    max_offset = _clamp(payload.get('max_offset_s', config['max_offset_s']), 0.0, 2.0, config['max_offset_s'])
    gate = _clamp(payload.get('gate_m', config['gate_m']), 0.2, 20.0, config['gate_m'])
    smooth = int(_clamp(payload.get('smooth_window', config['smooth_window']), 1, 15, config['smooth_window']))
    max_gap = _clamp(payload.get('max_gap_s', config['max_gap_s']), 0.0, 5.0, config['max_gap_s'])
    max_sample_gap = _clamp(payload.get('max_sample_gap_s', config['max_sample_gap_s']), 0.1, 3.0,
                            config['max_sample_gap_s'])

    if scenario == 'simulated':
        series, truth_t, truth_p, injected = simulate_series(config, payload)
    else:
        series = collect_live_series(config)
        truth_t = truth_p = injected = None

    cache_key = (scenario, round(hz, 3), round(gate, 3), smooth, round(max_gap, 3),
                 round(max_offset, 3), round(max_sample_gap, 3),
                 (payload.get('sim') or {}).get('seed', 42),
                 _data_signature(series))
    cached = _PIPELINE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if len(series) < 1:
        raise ValueError('没有可用的传感器轨迹数据')
    if len(series) == 1:
        scenario_label = '单传感器（仅同步与重建）'
    else:
        scenario_label = '多传感器协同' if scenario == 'live' else '多传感器协同（仿真）'

    sync = synchronize(series, hz=hz, max_offset=max_offset,
                       max_sample_gap=max_sample_gap)
    meta = {s.sensor_id: {'name': s.name, 'color': s.color, 'sigma': s.sigma} for s in series}
    fusion_result = fuse(sync.times, sync.aligned, sync.present, meta,
                         FusionConfig(gate_m=gate,
                                      sigmas={s.sensor_id: s.sigma for s in series},
                                      default_sigma=config['default_sigma']))
    recon = reconstruct(fusion_result.times, fusion_result.positions, fusion_result.sigmas,
                        ReconstructionConfig(max_gap_s=max_gap, smooth_window=smooth,
                                             outlier_sigma=config['outlier_sigma']))

    payload_out = {
        'success': True,
        'scenario': scenario,
        'scenario_label': scenario_label,
        'sensor_meta': meta,
        'sync': sync.metrics.to_dict(),
        'aligned': sync.to_dict(),
        'fusion': fusion_result.to_dict(),
        'reconstruction': recon.to_dict(),
        'reconstruction_metrics': recon.metrics,
        'injected': injected,
        'zone': config['zone'],
    }
    if truth_t is not None:
        payload_out['truth'] = {
            'times': [round(float(t), 4) for t in truth_t[::4]],
            'positions': [[round(float(v), 4) for v in p] for p in truth_p[::4]],
        }
        payload_out['accuracy'] = _accuracy(
            sync.times, sync.aligned, sync.present, fusion_result.positions,
            recon.points, truth_t, truth_p,
        )
        if injected:
            payload_out['injected'] = injected
            # 仿真注入的时钟偏置是相对真值的，而估计值是相对参考传感器的；
            # 这里把两者换算到同一基准，得到真实的偏置恢复误差。
            ref_id = sync.metrics.reference_id
            inj = injected.get('offsets') or {}
            base = float(inj.get(ref_id, 0.0))
            recovery = {}
            for sid, off in sync.metrics.offsets.items():
                if sid == ref_id:
                    continue
                recovery[sid] = abs(float(off) - (float(inj.get(sid, 0.0)) - base))
            payload_out['sync']['recovery_error'] = {k: round(v, 4) for k, v in recovery.items()}
    _cache_put(_PIPELINE_CACHE, cache_key, payload_out)
    return payload_out


# ── 预测与风险评估 ────────────────────────────────────

def _predict_future(times: np.ndarray, points: np.ndarray,
                    num_points: int, time_step: float) -> dict:
    """用 KinFlow 对重建轨迹做未来预测；模型不可用时退化为匀速外推。"""
    valid = np.isfinite(points).all(axis=1) & np.isfinite(times)
    t = times[valid]
    p = points[valid]
    if len(p) < 5:
        return {'times': [], 'positions': [], 'source': 'none'}

    try:
        from trajectory_reconstruction.core.prediction.prediction import (
            generate_kinflow_prediction, is_kinflow_model_available,
        )
        if is_kinflow_model_available():
            pred, pred_t = generate_kinflow_prediction(
                p.tolist(), t.tolist(), num_points=num_points, time_step=time_step,
                device=ensure_config().get('kinflow_model', {}).get('device', 'cpu'),
            )
            # 去掉锚点（第一个点即历史末点）
            return {
                'times': [round(float(x), 4) for x in pred_t[1:]],
                'positions': [[float(v) for v in q] for q in pred[1:]],
                'source': 'kinflow',
            }
    except Exception as exc:      # noqa: BLE001 - 预测失败不应阻断评估
        print(f'[评估] KinFlow 预测不可用: {exc}', flush=True)

    # 匀速外推兜底
    k = min(5, len(p))
    v = (p[-1] - p[-k]) / max(t[-1] - t[-k], 1e-6)
    out_t, out_p = [], []
    for i in range(1, num_points + 1):
        out_t.append(round(float(t[-1] + i * time_step), 4))
        out_p.append([float(x) for x in p[-1] + v * (i * time_step)])
    return {'times': out_t, 'positions': out_p, 'source': 'linear'}


def _zone_from(payload_zone: dict, config: dict) -> ProtectedZone:
    zone = dict(config.get('zone') or {})
    zone.update({k: v for k, v in (payload_zone or {}).items() if v is not None})
    center = zone.get('center', [8.0, 2.0, 3.0])
    if isinstance(center, (int, float)):
        center = [center, 2.0, 3.0]
    return ProtectedZone(
        center=(float(center[0]), float(center[1]), float(center[2])),
        radius=_clamp(zone.get('radius', 4.0), 0.5, 100.0, 4.0),
        y_min=_clamp(zone.get('y_min', 0.0), -100.0, 1000.0, 0.0),
        y_max=_clamp(zone.get('y_max', 20.0), -100.0, 5000.0, 20.0),
    )


def assess_scenario(payload: dict = None) -> dict:
    """完整评估: 重建 -> 预测 -> 风险分级 -> 处置建议。"""
    payload = payload or {}
    config = get_collab_config()

    # 数据版本也纳入缓存键，避免平台轨迹更新后仍返回旧评估
    data_stamp = _data_signature(collect_live_series(config))
    assess_key = (json.dumps(payload, sort_keys=True, ensure_ascii=False), data_stamp)
    cached = _ASSESS_CACHE.get(assess_key)
    if cached is not None:
        return cached

    pipeline = run_pipeline(payload)

    recon = pipeline['reconstruction']
    times = np.asarray(recon['times'], dtype=float)
    points = np.asarray(recon['points'], dtype=float)
    sigmas = np.asarray(recon['sigmas'], dtype=float)

    horizon = _clamp(payload.get('prediction_horizon_s', config['prediction_horizon_s']),
                     0.2, 10.0, config['prediction_horizon_s'])
    n_points = int(_clamp(payload.get('prediction_points', config['prediction_points']), 1, 30,
                          config['prediction_points']))
    step = round(horizon / max(n_points, 1), 4)
    prediction = _predict_future(times, points, n_points, step)

    zone = _zone_from(payload.get('zone'), config)
    weights = RiskWeights.from_dict(payload.get('weights') or config.get('weights'))
    thresholds = payload.get('thresholds') or config.get('thresholds')

    result = assess(
        times, points, zone=zone, prediction=prediction,
        uncertainty=sigmas, sensor_health=pipeline['fusion'].get('health'),
        weights=weights, thresholds=thresholds, horizon=horizon,
    )
    result['success'] = True
    result['scenario'] = pipeline['scenario']
    result['scenario_label'] = pipeline['scenario_label']
    result['prediction'] = prediction
    result['pipeline'] = {
        'sync': pipeline['sync'],
        'fusion': {k: pipeline['fusion'].get(k) for k in
                   ('health', 'agreement', 'track_jumps', 'n_sensors',
                    'selected', 'excluded')},
        'reconstruction': pipeline['reconstruction_metrics'],
        'accuracy': pipeline.get('accuracy'),
        'injected': pipeline.get('injected'),
    }
    result['sensor_meta'] = pipeline['sensor_meta']
    _cache_put(_ASSESS_CACHE, assess_key, result)
    return result


def sensor_status() -> list[dict]:
    """当前运行时各传感器的概览（供页面初始化）。"""
    config = get_collab_config()
    out = []
    for s in collect_live_series(config):
        dt = float(np.median(np.diff(s.times))) if len(s.times) > 1 else 0.0
        out.append({
            'id': s.sensor_id, 'name': s.name, 'color': s.color,
            'points': int(len(s.times)),
            'duration': round(float(s.times[-1] - s.times[0]), 2) if len(s.times) else 0.0,
            'rate_hz': round(1.0 / dt, 2) if dt > 0 else 0.0,
            'sigma': s.sigma,
        })
    return out
