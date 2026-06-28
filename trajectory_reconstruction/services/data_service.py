"""
数据管理服务 — 轨迹数据的加载、刷新、清理
"""
import os
from typing import Optional

from trajectory_reconstruction.core.state import detection_methods
from trajectory_reconstruction.core.config.config_manager import save_config
from trajectory_reconstruction.core.io.data_loader import load_dat_file, load_default_data


# ---------- 元数据持久化 ----------

def _get_metadata(methods: dict) -> dict:
    """从检测手段字典提取元数据（不含轨迹点）"""
    return {
        mid: {
            'name': data.get('name', ''),
            'color': data.get('color', '#999999'),
            'visible': data.get('visible', True),
        }
        for mid, data in methods.items()
    }


def save_metadata():
    """将当前检测手段的元信息持久化到 config.json"""
    from trajectory_reconstruction.core.config.config_manager import ensure_config
    cfg = ensure_config()
    cfg['detection_methods'] = _get_metadata(detection_methods)
    save_config(cfg)


# ---------- 数据操作 ----------

def initialize_data():
    """
    启动时初始化：从 data/fact/ 加载默认轨迹，移除自选平台
    """
    # 移除旧的自选平台
    if 'self' in detection_methods:
        del detection_methods['self']

    default_data = load_default_data()
    for mid, data in default_data.items():
        if mid in detection_methods:
            detection_methods[mid]['points'] = data['points']
            detection_methods[mid]['timestamps'] = data['timestamps']

    save_metadata()


def refresh_fact_data() -> bool:
    """
    重新加载 data/fact/*.dat 数据到 visible/infrared/radar
    保留 self（自选平台）不动
    返回是否有数据更新
    """
    default_data = load_default_data()
    updated = False
    for mid, data in default_data.items():
        if mid in detection_methods:
            detection_methods[mid]['points'] = data['points']
            detection_methods[mid]['timestamps'] = data['timestamps']
            updated = True
    save_metadata()
    return updated


def load_self_data(points: list, timestamps: list) -> dict:
    """
    创建/更新自选平台（self）的轨迹数据
    返回该平台的元信息
    """
    if 'self' not in detection_methods:
        detection_methods['self'] = {
            'name': '自选',
            'color': '#FF9500',  # Apple 风格橘色
            'visible': True,
            'points': [],
            'timestamps': [],
        }
    detection_methods['self']['points'] = points
    detection_methods['self']['timestamps'] = timestamps
    save_metadata()
    return {
        'name': detection_methods['self']['name'],
        'color': detection_methods['self']['color'],
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
