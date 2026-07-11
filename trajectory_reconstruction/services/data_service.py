"""
数据管理服务 — 轨迹数据的加载、刷新、清理
"""
import os
from typing import Optional

from trajectory_reconstruction.core.state import detection_methods
from trajectory_reconstruction.core.config.config_manager import save_config, ensure_config, PROJECT_ROOT
from trajectory_reconstruction.core.io.data_loader import (
    load_trajectory_file, save_trajectory_file, load_default_data,
)
from trajectory_reconstruction.core.math_utils import lerp_3d, nearest_avg_3d, smooth_points_3d

# ── 路径辅助 ──
def _fact_dir():
    return os.path.join(PROJECT_ROOT, 'data', 'fact')

def _predict_dir():
    return os.path.join(PROJECT_ROOT, 'data', 'predict')


# ---------- 元数据持久化 ----------

def _get_metadata(methods: dict) -> dict:
    """从检测手段字典提取元数据（不含轨迹点）"""
    result = {}
    for mid, data in methods.items():
        meta = {
            'name': data.get('name', ''),
            'color': data.get('color', '#999999'),
            'visible': data.get('visible', True),
            'enabled': data.get('enabled', True),
        }
        if 'weight' in data:
            meta['weight'] = data['weight']
        result[mid] = meta
    return result


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
    for ext in ('.npz', '.dat'):
        p = os.path.join(_fact_dir(), 'self' + ext)
        if os.path.isfile(p):
            os.remove(p)

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


def refresh_fact_data(keep_self: bool = False) -> bool:
    """
    重新加载 data/fact/ 数据到 visible/infrared/radar
    keep_self=True 时保留自选平台数据（恢复备份时使用）
    """
    default_data = load_default_data()
    updated = False
    for mid, data in default_data.items():
        if mid in detection_methods:
            detection_methods[mid]['points'] = data['points']
            detection_methods[mid]['timestamps'] = data['timestamps']
            updated = True

    if not keep_self:
        # 清除自选文件和数据
        for ext in ('.npz', '.dat'):
            p = os.path.join(_fact_dir(), 'self' + ext)
            if os.path.isfile(p):
                os.remove(p)
        if 'self' in detection_methods:
            detection_methods['self']['points'] = []
            detection_methods['self']['timestamps'] = []
    else:
        # 恢复时加载 self
        pts, tss = load_trajectory_file(os.path.join(_fact_dir(), 'self'))
        if pts:
            if 'self' not in detection_methods:
                detection_methods['self'] = {
                    'name': '自选', 'color': '#ff9500', 'visible': True, 'enabled': True,
                    'weight': 1.0, 'points': [], 'timestamps': [],
                }
            detection_methods['self']['points'] = pts
            detection_methods['self']['timestamps'] = tss
            updated = True

    # 清除所有预测文件
    _clear_predict_files()
    # 重新合成
    _auto_synthesize()
    save_metadata()
    return updated


def load_self_data(points: list, timestamps: list) -> dict:
    """
    创建/更新自选平台（self）的轨迹数据
    """
    if 'self' not in detection_methods:
        detection_methods['self'] = {
            'name': '自选', 'color': '#FF9500',
            'visible': True, 'enabled': True, 'weight': 1.0,
            'points': [], 'timestamps': [],
        }
    detection_methods['self']['enabled'] = True
    detection_methods['self']['visible'] = True
    detection_methods['self']['points'] = points
    detection_methods['self']['timestamps'] = timestamps
    save_trajectory_file(os.path.join(_fact_dir(), 'self'), points, timestamps)
    # 清除旧预测（数据已变，需重新预测）
    _clear_predict_files()
    save_metadata()
    _auto_synthesize()
    return {
        'name': detection_methods['self']['name'],
        'color': detection_methods['self']['color'],
    }


def _save_fact_file(filename: str, points: list, timestamps: list):
    """保存轨迹到 data/fact/ — 委托给 save_trajectory_file"""
    path = os.path.join(_fact_dir(), filename.replace('.dat', '').replace('.npz', ''))
    save_trajectory_file(path, points, timestamps)


def _clear_predict_files():
    """清除所有预测文件及综合轨迹内存数据"""
    pred_dir = _predict_dir()
    if os.path.isdir(pred_dir):
        for fname in os.listdir(pred_dir):
            fp = os.path.join(pred_dir, fname)
            if os.path.isfile(fp):
                os.remove(fp)
    # 清空综合轨迹（依赖预测数据，需一并清除）
    if 'synthetic' in detection_methods:
        detection_methods['synthetic']['points'] = []
        detection_methods['synthetic']['timestamps'] = []


def _auto_synthesize():
    """自动合成综合轨迹（静默忽略数据不足的情况）"""
    try:
        synthesize_trajectory()
    except ValueError:
        pass  # 数据不足时跳过，正常情况
    except Exception as e:
        print(f'[WARN] 综合轨迹合成异常: {e}')


def _ensure_synthetic_method():
    """确保综合平台存在于 detection_methods"""
    if 'synthetic' not in detection_methods:
        detection_methods['synthetic'] = {
            'name': '综合',
            'color': '#ffffff',
            'visible': True,
            'enabled': True,
            'points': [],
            'timestamps': [],
        }


def synthesize_trajectory() -> dict:
    """
    加权合成综合轨迹 — 合并所有启用的平台
    权重从各平台的 detection_methods[].weight 读取
    """
    active = {}
    for mid in ['visible', 'infrared', 'radar', 'self']:
        m = detection_methods.get(mid)
        if m and m.get('enabled', True) and m.get('points') and len(m['points']) >= 2:
            w = m.get('weight', 1.0)
            if w > 0:
                active[mid] = w

    if len(active) == 0:
        return {'success': False, 'error': '没有可用的平台'}

    # 构建统一时间网格（以各平台最小时间步为分辨率，覆盖交集区间）
    t_start = max(detection_methods[mid]['timestamps'][0] for mid in active)
    t_end = min(detection_methods[mid]['timestamps'][-1] for mid in active)
    if t_start >= t_end:
        return {'success': False, 'error': '平台时间范围无交集'}

    # 取最细时间步作为网格分辨率，保证不丢失运动细节
    min_dt = min(
        min((ts[i + 1] - ts[i]) for i in range(len(ts) - 1) if ts[i + 1] > ts[i])
        for mid in active
        if len(ts := detection_methods[mid].get('timestamps', [])) >= 2
    )
    if min_dt <= 0:
        min_dt = 0.1
    n_steps = max(2, int((t_end - t_start) / min_dt) + 1)
    sorted_ts = [round(t_start + i * (t_end - t_start) / (n_steps - 1), 6)
                 for i in range(n_steps)]

    # 加权平均
    syn_points, syn_times = [], []
    for t in sorted_ts:
        wx, wy, wz, ws = 0.0, 0.0, 0.0, 0.0
        for mid, weight in active.items():
            m_data = detection_methods[mid]
            p = lerp_3d(m_data['points'], m_data['timestamps'], t) or \
                nearest_avg_3d(m_data['points'], m_data['timestamps'], t)
            if p is not None:
                wx += p[0] * weight; wy += p[1] * weight; wz += p[2] * weight
                ws += weight
        if ws > 0:
            syn_points.append([wx/ws, wy/ws, wz/ws])
            syn_times.append(t)

    if len(syn_points) < 2:
        return {'success': False, 'error': '合成失败：有效插值点不足'}

    # 平滑（多平台融合才需要平滑，单平台直接使用原始点）
    if len(active) > 1:
        syn_points = smooth_points_3d(syn_points, passes=2)

    _ensure_synthetic_method()
    detection_methods['synthetic']['points'] = syn_points
    detection_methods['synthetic']['timestamps'] = syn_times
    detection_methods['synthetic']['visible'] = True
    save_metadata()

    return {
        'success': True,
        'point_count': len(syn_points),
        'platforms': list(active.keys()),
        'weights': dict(active),
    }


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
    fact_dir = _fact_dir()
    if os.path.isdir(fact_dir):
        for fname in os.listdir(fact_dir):
            fp = os.path.join(fact_dir, fname)
            if os.path.isfile(fp):
                os.remove(fp)

    # 3. 删除 predict/ 文件
    predict_dir = _predict_dir()
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
