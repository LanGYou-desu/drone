"""
数据文件 I/O — .dat 轨迹文件的读写操作

.dat 文件格式: 每行 `x y z t`，空格分隔
  x, y, z — 三维坐标（浮点数）
  t       — 时间戳（浮点数，秒）
"""
import os

from trajectory_reconstruction.core.config.config_manager import ensure_config


def load_dat_file(file_path: str) -> tuple[list[list[float]], list[float]]:
    """
    加载 .dat 轨迹文件

    Args:
        file_path: 文件路径

    Returns:
        points:     三维坐标列表 [[x, y, z], ...]
        timestamps: 时间戳列表 [t, ...]
        若文件不存在或格式异常，返回两个空列表
    """
    points: list[list[float]] = []
    timestamps: list[float] = []

    if not os.path.exists(file_path):
        print(f'[WARN] 文件不存在: {file_path}')
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


def save_predict_data(method_id: str, points: list[list[float]],
                      timestamps: list[float] | None = None):
    """
    将预测结果保存到 data/predict/pre{id}.dat
    """
    filename = f'predict_{method_id}.dat'
    file_path = os.path.join('data', 'predict', filename)

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    cfg = ensure_config()
    default_step = cfg.get('prediction_settings', {}).get('time_step', 0.5)

    with open(file_path, 'w', encoding='utf-8') as f:
        for i, p in enumerate(points):
            t = timestamps[i] if timestamps and i < len(timestamps) else i * default_step
            f.write(f'{p[0]} {p[1]} {p[2]} {t}\n')

    print(f'[OK] 预测数据已保存至 {file_path}')


def load_default_data() -> dict[str, dict[str, list]]:
    """
    加载 data/fact/ 下的默认轨迹数据

    Returns:
        { methodId: { 'points': [[x,y,z],...], 'timestamps': [t,...] } }
    """
    method_ids = ['visible', 'infrared', 'radar']
    file_names_new = ['visible.dat', 'infrared.dat', 'radar.dat']
    file_names_old = ['fact1.dat', 'fact2.dat', 'fact3.dat']  # 兼容旧备份

    methods: dict[str, dict[str, list]] = {}
    for mid, fname_new, fname_old in zip(method_ids, file_names_new, file_names_old):
        file_path = os.path.join('data', 'fact', fname_new)
        if not os.path.isfile(file_path):
            file_path = os.path.join('data', 'fact', fname_old)
        points, timestamps = load_dat_file(file_path)
        methods[mid] = {'points': points, 'timestamps': timestamps}

    return methods
