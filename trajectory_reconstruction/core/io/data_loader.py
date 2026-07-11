"""
数据文件 I/O — 轨迹文件的读写操作

支持格式:
  .npz (默认): np.savez(positions, timestamps) → data['positions'], data['timestamps']
  .dat (兼容): 每行 `x y z t`，空格分隔
"""

import os
import numpy as np

from trajectory_reconstruction.core.config.config_manager import ensure_config, _PROJECT_ROOT


def _data_dir(subdir: str = 'fact') -> str:
    """返回 data/ 子目录的绝对路径"""
    return os.path.join(_PROJECT_ROOT, 'data', subdir)


def load_dat_file(file_path: str) -> tuple[list[list[float]], list[float]]:
    """
    加载 .dat 文本轨迹文件（兼容旧格式）。

    返回 (points, timestamps)，文件不存在返回空列表。
    """
    points: list[list[float]] = []
    timestamps: list[float] = []

    if not os.path.exists(file_path):
        return points, timestamps

    try:
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
        print(f'[OK] 成功加载 {file_path}: {len(points)} 个轨迹点')
    except (ValueError, IOError) as e:
        print(f'[ERR] 加载文件失败 {file_path}: {e}')
        return [], []

    return points, timestamps


def load_npz_file(file_path: str) -> tuple[list[list[float]], list[float]]:
    """加载 .npz 轨迹文件，返回 (points, timestamps)"""
    points: list[list[float]] = []
    timestamps: list[float] = []

    if not os.path.exists(file_path):
        return points, timestamps

    try:
        data = np.load(file_path)
        pos = data['positions']
        ts = data['timestamps']
        points = pos.tolist()
        timestamps = ts.tolist()
        print(f'[OK] 成功加载 {file_path}: {len(points)} 个轨迹点')
    except Exception as e:
        print(f'[ERR] 加载 .npz 失败 {file_path}: {e}')
        return [], []

    return points, timestamps


def load_trajectory_file(base_path: str) -> tuple[list[list[float]], list[float]]:
    """
    加载轨迹文件，自动检测格式：.npz 优先，.dat 兜底。

    Args:
        base_path: 不含扩展名的文件路径，或完整 .npz/.dat 路径

    Returns:
        (points, timestamps)
    """
    # 如果已带扩展名，直接加载
    if base_path.endswith('.npz'):
        return load_npz_file(base_path)
    if base_path.endswith('.dat'):
        pts, ts = load_dat_file(base_path)
        if pts:
            return pts, ts
        # .dat 不存在时尝试 .npz
        npz_path = base_path[:-4] + '.npz'
        return load_npz_file(npz_path)

    # 不带扩展名：优先 .npz，回退 .dat
    npz_path = base_path + '.npz'
    if os.path.isfile(npz_path):
        return load_npz_file(npz_path)
    dat_path = base_path + '.dat'
    if os.path.isfile(dat_path):
        return load_dat_file(dat_path)

    return [], []


def save_trajectory_file(base_path: str, points: list[list[float]],
                         timestamps: list[float]):
    """保存轨迹为 .npz 格式"""
    path = base_path if base_path.endswith('.npz') else base_path + '.npz'
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    np.savez(path,
             positions=np.array(points, dtype=np.float32),
             timestamps=np.array(timestamps, dtype=np.float32))
    print(f'[OK] 轨迹已保存至 {path} ({len(points)} 个点)')


def save_predict_data(method_id: str, points: list[list[float]],
                      timestamps: list[float] | None = None):
    """将预测结果保存到 data/predict/{method_id}.npz"""
    filename = f'predict_{method_id}'
    file_path = os.path.join(_data_dir('predict'), filename)
    cfg = ensure_config()
    default_step = cfg.get('prediction_settings', {}).get('time_step', 0.5)

    if timestamps is None or len(timestamps) < len(points):
        timestamps = [i * default_step for i in range(len(points))]

    save_trajectory_file(file_path, points, list(timestamps))


def load_default_data() -> dict[str, dict[str, list]]:
    """
    加载 data/fact/ 下的默认轨迹数据（.npz 优先，.dat 兜底）。

    Returns:
        { methodId: { 'points': [[x,y,z],...], 'timestamps': [t,...] } }
    """
    method_ids = ['visible', 'infrared', 'radar']
    methods: dict[str, dict[str, list]] = {}
    fact_dir = _data_dir('fact')

    for mid in method_ids:
        base = os.path.join(fact_dir, mid)
        # 新格式(.npz)优先，旧格式(.dat)兜底
        points, timestamps = load_trajectory_file(base)

        # 兼容更旧的命名: fact1/2/3.dat
        if not points:
            old_map = {'visible': 'fact1', 'infrared': 'fact2', 'radar': 'fact3'}
            old_base = os.path.join(fact_dir, old_map[mid])
            points, timestamps = load_trajectory_file(old_base)

        methods[mid] = {'points': points, 'timestamps': timestamps}

    return methods
