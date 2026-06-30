"""
数据管理服务 — 轨迹数据的加载、刷新、清理
"""
import os
from typing import Optional

from trajectory_reconstruction.core.state import detection_methods
from trajectory_reconstruction.core.config.config_manager import save_config, ensure_config
from trajectory_reconstruction.core.io.data_loader import load_default_data


# ---------- 元数据持久化 ----------

def _get_metadata(methods: dict) -> dict:
    """从检测手段字典提取元数据（不含轨迹点）"""
    return {
        mid: {
            'name': data.get('name', ''),
            'color': data.get('color', '#999999'),
            'visible': data.get('visible', True),
            'weight': data.get('weight', 1.0),
        }
        for mid, data in methods.items()
    }


def save_metadata():
    """将当前检测手段的元信息持久化到 config.json"""
    cfg = ensure_config()
    cfg['detection_methods'] = _get_metadata(detection_methods)
    save_config(cfg)


# ---------- 数据操作 ----------

def initialize_data():
    """
    启动时初始化：从 data/fact/ 加载默认轨迹，清除预测文件
    """
    # 清除上次遗留的临时文件
    _clear_predict_files()
    self_path = os.path.join('data', 'fact', 'self.dat')
    if os.path.isfile(self_path):
        os.remove(self_path)

    default_data = load_default_data()
    for mid, data in default_data.items():
        if mid in detection_methods:
            detection_methods[mid]['points'] = data['points']
            detection_methods[mid]['timestamps'] = data['timestamps']

    # 清空自选平台数据
    if 'self' in detection_methods:
        detection_methods['self']['points'] = []
        detection_methods['self']['timestamps'] = []

    # 自动合成综合轨迹
    _auto_synthesize()

    save_metadata()


def refresh_fact_data() -> bool:
    """
    重新加载 data/fact/*.dat 数据到 visible/infrared/radar
    同时清空自选平台数据点（保留元信息）
    """
    default_data = load_default_data()
    updated = False
    for mid, data in default_data.items():
        if mid in detection_methods:
            detection_methods[mid]['points'] = data['points']
            detection_methods[mid]['timestamps'] = data['timestamps']
            updated = True
    # 清除自选文件和数据
    self_path = os.path.join('data', 'fact', 'self.dat')
    if os.path.isfile(self_path):
        os.remove(self_path)
    if 'self' in detection_methods:
        detection_methods['self']['points'] = []
        detection_methods['self']['timestamps'] = []
        updated = True
    # 清除所有预测文件
    _clear_predict_files()
    # 重新合成
    _auto_synthesize()
    save_metadata()
    return updated


def load_self_data(points: list, timestamps: list) -> dict:
    """
    创建/更新自选平台（self）的轨迹数据，持久化到 data/fact/self.dat
    """
    if 'self' not in detection_methods:
        detection_methods['self'] = {
            'name': '自选',
            'color': '#FF9500',
            'visible': True,
            'weight': 1.0,
            'points': [],
            'timestamps': [],
        }
    detection_methods['self']['points'] = points
    detection_methods['self']['timestamps'] = timestamps
    # 持久化
    _save_fact_file('self.dat', points, timestamps)
    save_metadata()
    _auto_synthesize()
    return {
        'name': detection_methods['self']['name'],
        'color': detection_methods['self']['color'],
    }


def _save_fact_file(filename: str, points: list, timestamps: list):
    """保存轨迹到 data/fact/ 目录"""
    os.makedirs(os.path.join('data', 'fact'), exist_ok=True)
    path = os.path.join('data', 'fact', filename)
    with open(path, 'w', encoding='utf-8') as f:
        for i, p in enumerate(points):
            t = timestamps[i] if i < len(timestamps) else i * 0.3
            f.write(f'{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {t:.6f}\n')


def _clear_predict_files():
    """清除所有预测文件"""
    pred_dir = os.path.join('data', 'predict')
    if os.path.isdir(pred_dir):
        for fname in os.listdir(pred_dir):
            fp = os.path.join(pred_dir, fname)
            if os.path.isfile(fp):
                os.remove(fp)


def _auto_synthesize():
    """自动合成综合轨迹（静默，忽略错误）"""
    try:
        synthesize_trajectory()
    except Exception:
        pass  # 数据不足时跳过


def _ensure_synthetic_method():
    """确保综合平台存在于 detection_methods"""
    if 'synthetic' not in detection_methods:
        detection_methods['synthetic'] = {
            'name': '综合',
            'color': '#ffffff',
            'visible': True,
            'points': [],
            'timestamps': [],
        }


def synthesize_trajectory() -> dict:
    """
    加权合成综合轨迹 — 合并所有有数据的平台
    权重从各平台的 detection_methods[].weight 读取
    """
    # 收集所有有数据的平台及其权重
    active = {}
    for mid in ['visible', 'infrared', 'radar', 'self']:
        m = detection_methods.get(mid)
        if m and m.get('points') and len(m['points']) >= 2:
            w = m.get('weight', 1.0)
            if w > 0:
                active[mid] = w

    if len(active) < 2:
        return {'success': False, 'error': '至少需要两个有数据的平台才能合成'}

    # 收集所有时间点并排序去重
    all_ts = set()
    for mid in active:
        ts = detection_methods[mid].get('timestamps', [])
        if ts:
            for t in ts:
                all_ts.add(round(t, 6))
    sorted_ts = sorted(all_ts)

    if len(sorted_ts) < 2:
        return {'success': False, 'error': '时间点不足'}

    # 对每个时间点，插值各平台坐标并加权平均
    syn_points = []
    syn_times = []
    for t in sorted_ts:
        wx_sum, wy_sum, wz_sum = 0.0, 0.0, 0.0
        w_sum = 0.0
        for mid, weight in active.items():
            p = _interpolate(mid, t) or _nearest_avg(mid, t)
            if p is not None:
                wx_sum += p[0] * weight
                wy_sum += p[1] * weight
                wz_sum += p[2] * weight
                w_sum += weight
        if w_sum > 0:
            syn_points.append([wx_sum / w_sum, wy_sum / w_sum, wz_sum / w_sum])
            syn_times.append(t)

    if len(syn_points) < 2:
        return {'success': False, 'error': '合成失败：有效插值点不足'}

    # 平滑处理（3点移动平均，重复2遍）
    for _ in range(2):
        smoothed = []
        for i in range(len(syn_points)):
            if i == 0 or i == len(syn_points) - 1:
                smoothed.append(syn_points[i][:])
            else:
                smoothed.append([
                    (syn_points[i-1][0] + syn_points[i][0] + syn_points[i+1][0]) / 3,
                    (syn_points[i-1][1] + syn_points[i][1] + syn_points[i+1][1]) / 3,
                    (syn_points[i-1][2] + syn_points[i][2] + syn_points[i+1][2]) / 3,
                ])
        syn_points = smoothed

    # 更新内存（综合轨迹不落盘，每次重新计算）
    _ensure_synthetic_method()
    detection_methods['synthetic']['points'] = syn_points
    detection_methods['synthetic']['timestamps'] = syn_times
    detection_methods['synthetic']['visible'] = True
    save_metadata()

    return {
        'success': True,
        'point_count': len(syn_points),
        'platforms': list(active.keys()),
        'weights': {k: v for k, v in active.items()},
    }


def _interpolate(mid: str, t: float):
    """在给定时间点线性插值某平台的坐标，无精确值时用前后平均"""
    m = detection_methods.get(mid)
    if not m:
        return None
    pts = m.get('points', [])
    ts = m.get('timestamps', [])
    if not pts or not ts or len(pts) < 2:
        return None

    if t <= ts[0]:
        return pts[0]
    if t >= ts[-1]:
        return pts[-1]

    import bisect
    i = bisect.bisect_left(ts, t)
    if i == 0:
        return pts[0]
    if i >= len(ts):
        return pts[-1]

    t0, t1 = ts[i-1], ts[i]
    if t1 == t0:
        return pts[i]

    ratio = (t - t0) / (t1 - t0)
    p0, p1 = pts[i-1], pts[i]
    return [
        p0[0] + (p1[0] - p0[0]) * ratio,
        p0[1] + (p1[1] - p0[1]) * ratio,
        p0[2] + (p1[2] - p0[2]) * ratio,
    ]


def _nearest_avg(mid: str, t: float):
    """前后最近两点的平均值（_interpolate 的兜底）"""
    m = detection_methods.get(mid)
    if not m:
        return None
    pts = m.get('points', [])
    ts = m.get('timestamps', [])
    if not pts or not ts or len(pts) < 2:
        return None
    # 找前后最近的点
    prev_pt, next_pt = None, None
    for i, ti in enumerate(ts):
        if ti <= t:
            prev_pt = pts[i]
        if ti >= t and next_pt is None:
            next_pt = pts[i]
    if prev_pt and next_pt:
        return [(prev_pt[j] + next_pt[j]) / 2 for j in range(3)]
    return prev_pt or next_pt


def clear_all_data() -> Optional[str]:
    """
    清理所有数据：
    1. 创建自动备份快照
    2. 删除 fact/ 和 predict/ 下的文件
    3. 清空内存轨迹
    返回快照名称
    """
    from trajectory_reconstruction.services.backup_service import create_backup

    # 1. 备份
    snapshot_name = create_backup(label='auto')

    # 2. 删除 fact/ 文件
    fact_dir = os.path.join('data', 'fact')
    if os.path.isdir(fact_dir):
        for fname in os.listdir(fact_dir):
            fp = os.path.join(fact_dir, fname)
            if os.path.isfile(fp):
                os.remove(fp)

    # 3. 删除 predict/ 文件
    predict_dir = os.path.join('data', 'predict')
    if os.path.isdir(predict_dir):
        for fname in os.listdir(predict_dir):
            fp = os.path.join(predict_dir, fname)
            if os.path.isfile(fp):
                os.remove(fp)

    # 4. 清空内存
    for mid in detection_methods:
        detection_methods[mid]['points'] = []
        detection_methods[mid]['timestamps'] = []

    save_metadata()
    return snapshot_name
