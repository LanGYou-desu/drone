from flask import Flask, render_template, jsonify, request
import config_manager
import ai_service
import prediction

app = Flask(__name__)

# 加载配置
config = config_manager.ensure_config()
detection_methods = config['detection_methods']
SF_API_KEY = config['siliconflow']['api_key']
SF_URL = config['siliconflow']['url']
SF_MODEL = config['siliconflow']['model']

@app.route('/')
def index():
    return render_template('index.html', methods_data=detection_methods)

@app.route('/api/ai_suggestion', methods=['POST'])
def ai_suggestion():
    data = request.json
    methods_data = data.get('methods_data', {})
    try:
        suggestion = ai_service.get_ai_suggestion(methods_data, SF_API_KEY, SF_URL, SF_MODEL)
        return jsonify({'success': True, 'suggestion': suggestion})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/predict', methods=['POST'])
def predict():
    """接收轨迹数据，返回预测点（可选）"""
    data = request.json
    points = data.get('points', [])
    num = data.get('num_points', 5)
    pred = prediction.generate_prediction(points, num)
    return jsonify({'success': True, 'prediction': pred})

@app.route('/api/refresh_data', methods=['POST'])
def refresh_data():
    """刷新数据（模拟），实际可从数据库加载"""
    # 这里简单模拟：对每个检测手段的点进行随机偏移
    import random
    for method_id in detection_methods:
        points = detection_methods[method_id]['points']
        new_points = []
        for p in points:
            new_points.append([
                p[0] + (random.random() - 0.5) * 0.3,
                p[1] + (random.random() - 0.5) * 0.2,
                p[2] + (random.random() - 0.5) * 0.3
            ])
        detection_methods[method_id]['points'] = new_points
    # 可选：保存到配置文件
    config_manager.save_config(config)
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)