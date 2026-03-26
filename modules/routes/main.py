"""
主页路由模块：提供主页渲染及所有 API 接口
轨迹数据从文件加载，不写入 config.json
"""
import os
from flask import Blueprint, render_template, jsonify, request
from modules.data.data_loader import load_dat_file, save_predict_data, load_default_data, backup_data, clear_all_data
from modules.predict.prediction import generate_prediction
from modules.ai.ai_service import get_ai_suggestion
from modules.config.config_manager import ensure_config, save_config

main_bp = Blueprint('main', __name__)

# 加载配置（仅用于 API 密钥和检测手段元信息）
config = ensure_config()
detection_methods = config['detection_methods']  # 包含名称、颜色、可见性
SF_API_KEY = config['siliconflow']['api_key']
SF_URL = config['siliconflow']['url']
SF_MODEL = config['siliconflow']['model']

# ---------- 辅助函数：仅保存元数据到 config.json ----------
def _get_metadata(methods):
    metadata = {}
    for mid, data in methods.items():
        metadata[mid] = {
            'name': data.get('name', ''),
            'color': data.get('color', '#ffffff'),
            'visible': data.get('visible', True)
        }
    return metadata

def _save_config_metadata(methods):
    metadata = _get_metadata(methods)
    full_config = ensure_config()
    full_config['detection_methods'] = metadata
    save_config(full_config)

# -------------------------------------------------------

def initialize_data():
    """从 data/fact/ 加载默认轨迹数据到内存，并打印加载结果"""
    default_data = load_default_data()
    for mid, data in default_data.items():
        if mid in detection_methods:
            detection_methods[mid]['points'] = data['points']
            detection_methods[mid]['timestamps'] = data['timestamps']
            print(f"📊 {mid}: 加载了 {len(data['points'])} 个轨迹点")
        else:
            print(f"⚠️ 检测手段 {mid} 不存在于配置中")
    _save_config_metadata(detection_methods)

initialize_data()

@main_bp.route('/')
def index():
    return render_template('index.html', methods_data=detection_methods)

@main_bp.route('/api/ai_suggestion', methods=['POST'])
def ai_suggestion():
    data = request.json
    methods_data = data.get('methods_data', {})
    try:
        suggestion = get_ai_suggestion(methods_data, SF_API_KEY, SF_URL, SF_MODEL)
        return jsonify({'success': True, 'suggestion': suggestion})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@main_bp.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    method_id = data.get('method_id')
    points = data.get('points', [])
    timestamps = data.get('timestamps', [])
    num_points = data.get('num_points', 6)
    time_step = data.get('time_step', 0.5)

    if method_id == 'all':
        # 对所有可见平台进行预测
        results = {}
        for mid, info in detection_methods.items():
            if info.get('visible', True) and info.get('points') and len(info['points']) >= 2:
                pred_points, pred_times = generate_prediction(
                    info['points'], info.get('timestamps', []), num_points, time_step
                )
                if pred_points:
                    save_predict_data(mid, pred_points, pred_times)
                    results[mid] = {'prediction': pred_points, 'pred_times': pred_times}
        return jsonify({'success': True, 'results': results})
    else:
        pred_points, pred_times = generate_prediction(points, timestamps, num_points, time_step)
        if pred_points:
            save_predict_data(method_id, pred_points, pred_times)
        return jsonify({'success': True, 'prediction': pred_points, 'pred_times': pred_times})

@main_bp.route('/api/load_data', methods=['POST'])
def load_data():
    file = request.files.get('file')
    method_id = request.form.get('method_id')
    if not file or not method_id:
        return jsonify({'success': False, 'error': 'No file or method_id'})

    points, timestamps = load_dat_file(file)
    if not points:
        return jsonify({'success': False, 'error': 'Invalid file format'})

    if method_id == 'self':
        if 'self' not in detection_methods:
            detection_methods['self'] = {
                'name': '自选',
                'color': '#ff8c42',
                'visible': True,
                'points': [],
                'timestamps': []
            }
        detection_methods['self']['points'] = points
        detection_methods['self']['timestamps'] = timestamps
        _save_config_metadata(detection_methods)
        return jsonify({'success': True, 'method_id': method_id, 'name': detection_methods[method_id]['name']})
    else:
        return jsonify({'success': False, 'error': 'Invalid method_id'})

@main_bp.route('/api/clear_data', methods=['POST'])
def clear_data():
    """
    清空所有数据：备份当前 data 目录，然后清空 data/fact 和 data/predict 内容，
    并重置内存中的检测手段轨迹点
    """
    try:
        # 1. 备份当前数据
        backup_data()
        # 2. 清空文件系统中的数据
        clear_all_data()
        # 3. 重置内存中的轨迹数据
        for mid in detection_methods:
            detection_methods[mid]['points'] = []
            detection_methods[mid]['timestamps'] = []
        # 4. 保存元数据到 config
        _save_config_metadata(detection_methods)
        return jsonify({'success': True, 'message': '数据已清空，备份保存在 data/backup'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@main_bp.route('/api/refresh_data', methods=['POST'])
def refresh_data():
    """重新加载默认 data/fact/*.dat 文件，清除自选数据"""
    default_data = load_default_data()
    for mid, data in default_data.items():
        detection_methods[mid]['points'] = data['points']
        detection_methods[mid]['timestamps'] = data['timestamps']
    if 'self' in detection_methods:
        detection_methods['self']['points'] = []
        detection_methods['self']['timestamps'] = []
    _save_config_metadata(detection_methods)
    return jsonify({'success': True})