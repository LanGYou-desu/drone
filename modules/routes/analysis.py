"""
分析页面路由：提供详细数据分析页面及其数据接口
"""
from flask import Blueprint, render_template, jsonify
from modules.routes.main import detection_methods   # 共享同一个全局对象

analysis_bp = Blueprint('analysis', __name__)


@analysis_bp.route('/')
def analysis():
    """渲染分析页面，并将检测手段元数据传递给模板"""
    # 构建只含元数据的字典，避免将大量轨迹数据传到模板
    metadata = {}
    for mid, data in detection_methods.items():
        metadata[mid] = {
            'name': data.get('name', ''),
            'color': data.get('color', '#ffffff'),
            'visible': data.get('visible', True)
        }
    return render_template('analysis.html', methods_data=metadata)


@analysis_bp.route('/data')
def get_analysis_data():
    """
    返回各检测手段的详细分析数据：
    - 高度变化 (heights)
    - 速度变化 (speeds)
    - 加速度变化 (accelerations)
    - 曲率变化 (curvatures)
    - 对应的时间戳 (time_steps)
    """
    print("=== Analysis Data Request ===")
    result = {}
    for method_id, data in detection_methods.items():
        points = data.get('points', [])
        timestamps = data.get('timestamps', [])
        print(f"{method_id}: points={len(points)}, timestamps={len(timestamps)}")

        # 若无有效点，返回空数据
        if len(points) < 2:
            result[method_id] = {
                'name': data['name'],
                'color': data['color'],
                'speeds': [],
                'accelerations': [],
                'curvatures': [],
                'heights': [p[1] for p in points],
                'time_steps': timestamps
            }
            continue

        # ----- 1. 速度 (基于水平位移和实际时间差) -----
        speeds = []
        for i in range(1, len(points)):
            dt = timestamps[i] - timestamps[i-1] if timestamps and i < len(timestamps) else 1.0
            if dt == 0:
                dt = 1.0
            dx = points[i][0] - points[i-1][0]
            dz = points[i][2] - points[i-1][2]
            speed = (dx**2 + dz**2)**0.5 / dt
            speeds.append(speed)

        # ----- 2. 加速度 (速度变化率) -----
        accelerations = []
        for i in range(1, len(speeds)):
            accelerations.append(speeds[i] - speeds[i-1])

        # ----- 3. 曲率 (近似计算平面曲率) -----
        curvatures = []
        for i in range(1, len(points)-1):
            p0 = points[i-1]
            p1 = points[i]
            p2 = points[i+1]

            d1 = [p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]]
            d2 = [p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]]

            cross = [d1[1]*d2[2] - d1[2]*d2[1],
                     d1[2]*d2[0] - d1[0]*d2[2],
                     d1[0]*d2[1] - d1[1]*d2[0]]
            cross_norm = (cross[0]**2 + cross[1]**2 + cross[2]**2)**0.5
            d1_norm = (d1[0]**2 + d1[1]**2 + d1[2]**2)**0.5
            curvature = cross_norm / (d1_norm**3) if d1_norm > 0 else 0
            curvatures.append(curvature)

        result[method_id] = {
            'name': data['name'],
            'color': data['color'],
            'speeds': speeds,
            'accelerations': accelerations,
            'curvatures': curvatures,
            'heights': [p[1] for p in points],
            'time_steps': timestamps
        }
    print(f"Generated result for {len(result)} methods.")
    return jsonify(result)