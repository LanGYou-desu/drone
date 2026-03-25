from flask import Blueprint, render_template, jsonify
import data_manager

analysis_bp = Blueprint('analysis', __name__)

@analysis_bp.route('/')
def analysis():
    detection_methods = data_manager.load_methods()
    return render_template('analysis.html', methods_data=detection_methods)

@analysis_bp.route('/data')
def get_analysis_data():
    methods = data_manager.load_methods()
    result = {}
    for method_id, data in methods.items():
        points = data['points']
        if len(points) < 2:
            result[method_id] = {
                'name': data['name'],
                'color': data['color'],
                'speeds': [],
                'accelerations': [],
                'curvatures': [],
                'heights': [p[1] for p in points],
                'time_steps': list(range(len(points)))
            }
            continue

        # 速度 (假设时间步长1)
        speeds = []
        for i in range(1, len(points)):
            dx = points[i][0] - points[i-1][0]
            dz = points[i][2] - points[i-1][2]
            speed = (dx**2 + dz**2)**0.5
            speeds.append(speed)

        # 加速度
        accelerations = []
        for i in range(1, len(speeds)):
            accelerations.append(speeds[i] - speeds[i-1])

        # 曲率
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
            'time_steps': list(range(len(points)))
        }
    return jsonify(result)