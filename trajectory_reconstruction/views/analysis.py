"""
分析页面路由 — 提供数据分析页面及原始数据接口（含预测数据）
"""
import math
import os
from flask import Blueprint, render_template, jsonify

from trajectory_reconstruction.core.state import detection_methods, METHOD_ORDER
from trajectory_reconstruction.core.io.data_loader import load_trajectory_file, _data_dir
from trajectory_reconstruction.core.config.config_manager import ensure_config, PROJECT_ROOT

analysis_bp = Blueprint('analysis', __name__)

# 预测文件映射
_PREDICT_FILE_MAP = {
    'visible': 'predict_visible', 'infrared': 'predict_infrared',
    'radar': 'predict_radar', 'self': 'predict_self', 'synthetic': 'predict_synthetic',
}


def _load_predict_data(method_id: str):
    """加载某平台的预测数据（若存在）"""
    fname = _PREDICT_FILE_MAP.get(method_id, f'predict_{method_id}')
    file_path = os.path.join(_data_dir('predict'), fname)
    return load_trajectory_file(file_path)


def _smooth_points(points, window: int = 5):
    """
    位置序列滑动平均（端点保持）。

    传感器原始观测含 0.25~0.6 m 测量噪声，直接差分求速度/加速度会把噪声放大
    1/Δt 倍（4 Hz 下相当于 2 m/s 量级），曲线完全被噪声淹没；先做 5 点平滑
    再差分，得到的是"可读的运动趋势"，与界面上"分析"的定位一致。
    """
    n = len(points)
    if n < 3 or window <= 1:
        return list(points)
    half = window // 2
    out = []
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        chunk = points[lo:hi]
        out.append([sum(p[a] for p in chunk) / len(chunk) for a in range(3)])
    return out


def _compute_metrics(points, timestamps):
    """计算运动学指标（先平滑再差分，加速度按 m/s² 计算）"""
    if len(points) < 2:
        return [], [], [], [p[1] for p in points]

    smooth = _smooth_points(points, 5)

    speeds = []
    step_dt = []
    for i in range(1, len(smooth)):
        dt = timestamps[i] - timestamps[i - 1] if i < len(timestamps) else 1.0
        if dt <= 0:
            dt = 1.0
        step_dt.append(dt)
        dx = smooth[i][0] - smooth[i - 1][0]
        dz = smooth[i][2] - smooth[i - 1][2]
        speeds.append((dx ** 2 + dz ** 2) ** 0.5 / dt)

    accelerations = []
    for i in range(1, len(speeds)):
        dt = step_dt[i] if i < len(step_dt) else 1.0
        accelerations.append((speeds[i] - speeds[i - 1]) / max(dt, 1e-3))

    curvatures = []
    for i in range(1, len(smooth) - 1):
        p0, p1, p2 = smooth[i - 1], smooth[i], smooth[i + 1]
        d1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]]
        d2 = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]]
        cross = [
            d1[1] * d2[2] - d1[2] * d2[1],
            d1[2] * d2[0] - d1[0] * d2[2],
            d1[0] * d2[1] - d1[1] * d2[0],
        ]
        cross_norm = math.sqrt(cross[0]**2 + cross[1]**2 + cross[2]**2)
        d1_norm = math.sqrt(d1[0]**2 + d1[1]**2 + d1[2]**2)
        curvatures.append(cross_norm / (d1_norm ** 3) if d1_norm > 1e-6 else 0.0)

    return speeds, accelerations, curvatures, [p[1] for p in points]


@analysis_bp.route('/')
def analysis_page():
    """分析评估页面（运动学分析 + 智能风险评估）"""
    metadata = {}
    for mid, data in detection_methods.items():
        if data.get('visible', True):
            metadata[mid] = {
                'name': data.get('name', ''),
                'color': data.get('color', '#999999'),
                'visible': True,
            }
    from trajectory_reconstruction.services.sensor_service import get_collab_config
    collab = get_collab_config()
    return render_template('analysis.html', methods_data=metadata, active_page='analysis',
                           collab_config=collab, zone=collab.get('zone', {}))


@analysis_bp.route('/data')
def get_analysis_data():
    """返回各检测手段的运动学分析数据（fact + predict 拼接后统一计算）"""
    self_exists = os.path.isfile(os.path.join(PROJECT_ROOT, 'data', 'fact', 'self.npz')) or \
                  os.path.isfile(os.path.join(PROJECT_ROOT, 'data', 'fact', 'self.dat'))
    result = {}
    for mid in METHOD_ORDER:
        if mid not in detection_methods:
            continue
        if mid == 'self' and not self_exists:
            continue
        data = detection_methods[mid]
        if not data.get('visible', True) or len(data.get('points', [])) < 2:
            continue
        # 拼接原始数据和预测数据
        points = list(data.get('points', []))
        timestamps = list(data.get('timestamps', []))
        pred_pts, pred_ts = _load_predict_data(mid)
        if pred_pts and len(pred_pts) >= 2:
            if points and abs(pred_ts[0] - timestamps[-1]) < 1e-6:
                points.extend(pred_pts[1:])
                timestamps.extend(pred_ts[1:])
            else:
                points.extend(pred_pts)
                timestamps.extend(pred_ts)

        speeds, accelerations, curvatures, heights = _compute_metrics(points, timestamps)

        # 对齐数组长度：speed 少 1 个点，accel/curv 各少 2 个点，补前导零
        speeds = [0] + speeds
        accelerations = [0, 0] + accelerations
        curvatures = [0] + curvatures + [0]

        result[mid] = {
            'name': data['name'],
            'color': data['color'],
            'speeds': speeds,
            'accelerations': accelerations,
            'curvatures': curvatures,
            'heights': heights,
            'time_steps': timestamps,
        }

    # METHOD_ORDER 优先，其余追加
    sorted_result = {mid: result[mid] for mid in METHOD_ORDER if mid in result}
    sorted_result.update({mid: result[mid] for mid in result if mid not in sorted_result})
    return jsonify(sorted_result)


@analysis_bp.route('/capture')
def get_capture_analysis():
    """返回综合轨迹最佳捕捉时机（仅预测后可用）"""
    mid = 'synthetic'
    if mid not in detection_methods:
        return jsonify([])
    data = detection_methods[mid]
    if not data.get('visible', True) or len(data.get('points', [])) < 2:
        return jsonify([])

    pred_pts, pred_ts = _load_predict_data(mid)
    if not pred_pts or len(pred_pts) < 3:
        return jsonify([])

    cfg = ensure_config()
    w = cfg.get('capture_weights', {'height': 0.3, 'speed': 0.3, 'acceleration': 0.2, 'curvature': 0.2})
    w_h, w_s, w_a, w_c = w['height'], w['speed'], w['acceleration'], w['curvature']
    # 归一化权重使总和为 1，保证最高分 ≤ 100
    w_sum = w_h + w_s + w_a + w_c
    if w_sum > 0:
        w_h, w_s, w_a, w_c = w_h / w_sum, w_s / w_sum, w_a / w_sum, w_c / w_sum

    n = len(pred_pts)
    if n < 4:
        return jsonify([])

    heights = [p[1] for p in pred_pts]
    speeds, accels, curvs, _ = _compute_metrics(pred_pts, pred_ts)
    # 对齐到点索引；缺失（首/末点无法差分）记为 None，按中性值处理，
    # 避免"锚点速度为 0"被误判成最优捕捉点。
    speed_arr = [None] + list(speeds)
    accel_arr = [None, None] + list(accels)
    curv_arr = [None] + list(curvs) + [None]

    def _vmax(arr):
        vals = [abs(v) for v in arr if v is not None]
        return max(vals) if vals else 1.0

    h_max = max(heights) or 1
    s_max = _vmax(speed_arr) or 1
    a_max = _vmax(accel_arr) or 1
    c_max = _vmax(curv_arr) or 1

    def _norm(v, vmax):
        if v is None:
            return 0.5          # 缺失指标按中性处理，不奖励也不惩罚
        return max(0.0, min(1.0, 1.0 - abs(v) / vmax))

    scored = []
    # 只在预测段内部取候选：跳过锚点（当前时刻）与末端外推点
    for i in range(1, n - 1):
        sc = (w_h * (1 - heights[i] / h_max) +
              w_s * _norm(speed_arr[i], s_max) +
              w_a * _norm(accel_arr[i], a_max) +
              w_c * _norm(curv_arr[i], c_max))
        scored.append((sc, i))
    if not scored:
        return jsonify([])
    # 以最高分为 100，其余按比例缩放
    max_sc = max(s for s, _ in scored)
    if max_sc > 0:
        scored = [(round(s / max_sc * 100), i) for s, i in scored]
    scored.sort(reverse=True)
    top3 = scored[:3]

    return jsonify([{
        'rank': j + 1,
        'time': round(pred_ts[i], 2),
        'position': [round(pred_pts[i][0], 2), round(pred_pts[i][1], 2), round(pred_pts[i][2], 2)],
        'score': s,
        'height': round(heights[i], 2),
        'speed': round(speed_arr[i] or 0.0, 2),
    } for j, (s, i) in enumerate(top3)])
