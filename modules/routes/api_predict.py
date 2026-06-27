"""
预测 API — 单平台/全平台轨迹预测 + AI 建议
"""
from flask import Blueprint, jsonify, request

from modules.services import predict_service
from modules.services.state import detection_methods, SF_API_KEY, SF_URL, SF_MODEL
from modules.ai.ai_service import get_ai_suggestion

predict_bp = Blueprint('predict', __name__)


@predict_bp.route('/api/predict', methods=['POST'])
def predict():
    """
    对指定平台进行轨迹预测
    Body: { method_id, points, timestamps?, num_points?, time_step? }
    """
    data = request.get_json(silent=True) or {}
    method_id = data.get('method_id', '')
    num_points = data.get('num_points', 6)
    time_step = data.get('time_step')

    if not method_id:
        return jsonify({'success': False, 'error': '缺少 method_id'}), 400

    if method_id not in detection_methods:
        return jsonify({'success': False, 'error': f'未知平台: {method_id}'}), 400

    result = predict_service.predict_single(method_id, num_points, time_step)
    if result is None:
        return jsonify({'success': False, 'error': '该平台数据不足，无法预测'}), 400

    return jsonify({
        'success': True,
        'prediction': result['prediction'],
        'pred_times': result['pred_times'],
    })


@predict_bp.route('/api/predict_all', methods=['POST'])
def predict_all():
    """
    对所有可见平台进行预测
    Body: { num_points?, time_step? }
    """
    data = request.get_json(silent=True) or {}
    num_points = data.get('num_points', 6)
    time_step = data.get('time_step')

    results = predict_service.predict_all(num_points, time_step)
    return jsonify({'success': True, 'results': results})


@predict_bp.route('/api/ai_suggestion', methods=['POST'])
def ai_suggestion():
    """
    获取 AI 无人机捕捉策略建议
    Body: { methods_data: { methodId: { name, points, timestamps } } }
    """
    data = request.get_json(silent=True) or {}
    methods_data = data.get('methods_data', {})

    if not methods_data:
        return jsonify({'success': False, 'error': '缺少 methods_data'}), 400

    try:
        suggestion = get_ai_suggestion(methods_data, SF_API_KEY, SF_URL, SF_MODEL)
        return jsonify({'success': True, 'suggestion': suggestion})
    except Exception as e:
        return jsonify({'success': False, 'error': f'AI 服务异常: {str(e)}'}), 500
