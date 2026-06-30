"""
分析页面路由 — 提供数据分析页面及原始数据接口（含预测数据）
"""
import os
from flask import Blueprint, render_template, jsonify

from trajectory_reconstruction.core.state import detection_methods, METHOD_ORDER
from trajectory_reconstruction.core.io.data_loader import load_dat_file
from trajectory_reconstruction.core.config.config_manager import ensure_config

analysis_bp = Blueprint('analysis', __name__)

# 预测文件映射
_PREDICT_FILE_MAP = {'visible': 'pre1.dat', 'infrared': 'pre2.dat', 'radar': 'pre3.dat',
                     'self': 'preself.dat', 'synthetic': 'presyn.dat'}


def _load_predict_data(method_id: str):
    """加载某平台的预测数据（若存在）"""
    fname = _PREDICT_FILE_MAP.get(method_id, f'pre{method_id}.dat')
    file_path = os.path.join('data', 'predict', fname)
    return load_dat_file(file_path)


def _compute_metrics(points, timestamps):
    """计算运动学指标"""
    if len(points) < 2:
        return [], [], [], [p[1] for p in points]

    speeds = []
    for i in range(1, len(points)):
        dt = timestamps[i] - timestamps[i - 1] if i < len(timestamps) else 1.0
        if dt <= 0:
            dt = 1.0
        dx = points[i][0] - points[i - 1][0]
        dz = points[i][2] - points[i - 1][2]
        speeds.append((dx ** 2 + dz ** 2) ** 0.5 / dt)

    accelerations = []
    for i in range(1, len(speeds)):
        accelerations.append(speeds[i] - speeds[i - 1])

    curvatures = []
    for i in range(1, len(points) - 1):
        p0, p1, p2 = points[i - 1], points[i], points[i + 1]
        d1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]]
        d2 = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]]
        cross = [
            d1[1] * d2[2] - d1[2] * d2[1],
            d1[2] * d2[0] - d1[0] * d2[2],
            d1[0] * d2[1] - d1[1] * d2[0],
        ]
        cross_norm = (cross[0] ** 2 + cross[1] ** 2 + cross[2] ** 2) ** 0.5
        d1_norm = (d1[0] ** 2 + d1[1] ** 2 + d1[2] ** 2) ** 0.5
        curvatures.append(cross_norm / d1_norm ** 3 if d1_norm > 0 else 0)

    return speeds, accelerations, curvatures, [p[1] for p in points]


@analysis_bp.route('/')
def analysis_page():
    """分析页面"""
    metadata = {}
    for mid, data in detection_methods.items():
        if data.get('visible', True):
            metadata[mid] = {
                'name': data.get('name', ''),
                'color': data.get('color', '#999999'),
                'visible': True,
            }
    return render_template('analysis.html', methods_data=metadata, active_page='analysis')


@analysis_bp.route('/data')
def get_analysis_data():
    """返回各检测手段的运动学分析数据（fact + predict 拼接后统一计算）"""
    self_exists = os.path.isfile(os.path.join('data', 'fact', 'self.dat'))
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

    heights = [p[1] for p in pred_pts]
    speeds, accels, curvs, _ = _compute_metrics(pred_pts, pred_ts)
    speeds = [0] + speeds
    accels = [0, 0] + accels
    curvs = [0] + curvs + [0]

    n = len(pred_pts)
    def _safe(arr, i): return arr[i] if i < len(arr) else 0
    h_max = max(heights) or 1
    s_max = max(speeds) or 1
    a_max = max((abs(v) for v in accels), default=1) or 1
    c_max = max(curvs) or 1

    scored = []
    for i in range(n):
        sc = (w_h * (1 - _safe(heights, i) / h_max) +
              w_s * (1 - _safe(speeds, i) / s_max) +
              w_a * (1 - abs(_safe(accels, i)) / a_max) +
              w_c * (1 - _safe(curvs, i) / c_max))
        scored.append((sc, i))
    # 以最高分为 100，其余按比例缩放
    max_sc = max(s for s, _ in scored) if scored else 1
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
        'speed': round(speeds[i], 2),
    } for j, (s, i) in enumerate(top3)])
