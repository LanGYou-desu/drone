"""
分析页面路由 — 提供数据分析页面及原始数据接口
"""
from flask import Blueprint, render_template, jsonify

from trajectory_reconstruction.core.state import detection_methods

analysis_bp = Blueprint('analysis', __name__)


@analysis_bp.route('/')
def analysis_page():
    """分析页面"""
    metadata = {}
    for mid, data in detection_methods.items():
        metadata[mid] = {
            'name': data.get('name', ''),
            'color': data.get('color', '#999999'),
            'visible': data.get('visible', True),
        }
    return render_template('analysis.html', methods_data=metadata, active_page='analysis')


@analysis_bp.route('/data')
def get_analysis_data():
    """
    返回各检测手段的详细运动学分析数据:
      heights       — 高度序列
      speeds        — 速度序列 (水平位移 / 时间差)
      accelerations — 加速度序列 (速度变化率)
      curvatures    — 曲率序列 (三点法平面曲率)
      time_steps    — 时间戳序列
    """
    result = {}
    for method_id, data in detection_methods.items():
        points = data.get('points', [])
        timestamps = data.get('timestamps', [])

        # 无有效数据
        if len(points) < 2:
            result[method_id] = {
                'name': data['name'],
                'color': data['color'],
                'speeds': [],
                'accelerations': [],
                'curvatures': [],
                'heights': [p[1] for p in points],
                'time_steps': timestamps,
            }
            continue

        # 速度
        speeds = []
        for i in range(1, len(points)):
            dt = timestamps[i] - timestamps[i - 1] if timestamps and i < len(timestamps) else 1.0
            if dt <= 0:
                dt = 1.0
            dx = points[i][0] - points[i - 1][0]
            dz = points[i][2] - points[i - 1][2]
            speeds.append((dx ** 2 + dz ** 2) ** 0.5 / dt)

        # 加速度
        accelerations = []
        for i in range(1, len(speeds)):
            accelerations.append(speeds[i] - speeds[i - 1])

        # 曲率（三点法）
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

        result[method_id] = {
            'name': data['name'],
            'color': data['color'],
            'speeds': speeds,
            'accelerations': accelerations,
            'curvatures': curvatures,
            'heights': [p[1] for p in points],
            'time_steps': timestamps,
        }

    return jsonify(result)
