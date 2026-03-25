from flask import Blueprint, render_template, jsonify, request
import data_manager
import prediction
import ai_service
from config_manager import ensure_config

main_bp = Blueprint('main', __name__)

config = ensure_config()
detection_methods = data_manager.load_methods()
SF_API_KEY = config['siliconflow']['api_key']
SF_URL = config['siliconflow']['url']
SF_MODEL = config['siliconflow']['model']

@main_bp.route('/')
def index():
    return render_template('index.html', methods_data=detection_methods)

@main_bp.route('/api/ai_suggestion', methods=['POST'])
def ai_suggestion():
    data = request.json
    methods_data = data.get('methods_data', {})
    try:
        suggestion = ai_service.get_ai_suggestion(methods_data, SF_API_KEY, SF_URL, SF_MODEL)
        return jsonify({'success': True, 'suggestion': suggestion})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@main_bp.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    points = data.get('points', [])
    num = data.get('num_points', 5)
    pred = prediction.generate_prediction(points, num)
    return jsonify({'success': True, 'prediction': pred})

@main_bp.route('/api/refresh_data', methods=['POST'])
def refresh_data():
    import random
    methods = data_manager.load_methods()
    for method_id in methods:
        points = methods[method_id]['points']
        new_points = []
        for p in points:
            new_points.append([
                p[0] + (random.random() - 0.5) * 0.3,
                p[1] + (random.random() - 0.5) * 0.2,
                p[2] + (random.random() - 0.5) * 0.3
            ])
        methods[method_id]['points'] = new_points
    data_manager.save_methods(methods)
    return jsonify({'success': True})