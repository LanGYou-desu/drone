"""
主页路由模块：提供主页渲染及所有 API 接口
轨迹数据从文件加载，不写入 config.json
"""
import os
import shutil
import time
from flask import Blueprint, render_template, jsonify, request
from modules.data.data_loader import load_dat_file, save_predict_data, load_default_data
from modules.predict.prediction import generate_prediction
from modules.ai.ai_service import get_ai_suggestion
from modules.config.config_manager import ensure_config, save_config

main_bp = Blueprint('main', __name__)

# 加载配置（仅用于 API 密钥和检测手段元信息）
config = ensure_config()
detection_methods = config['detection_methods']
SF_API_KEY = config['siliconflow']['api_key']
SF_URL = config['siliconflow']['url']
SF_MODEL = config['siliconflow']['model']

# ---------- 辅助函数：仅保存元数据到 config.json ----------
def _get_metadata(methods):
    """从完整的检测手段字典中提取仅包含元数据的部分（不含轨迹）"""
    metadata = {}
    for mid, data in methods.items():
        metadata[mid] = {
            'name': data.get('name', ''),
            'color': data.get('color', '#ffffff'),
            'visible': data.get('visible', True)
        }
    return metadata

def _save_config_metadata(methods):
    """只保存元数据到配置文件，保留其他配置项（如 API 密钥）"""
    metadata = _get_metadata(methods)
    full_config = ensure_config()
    full_config['detection_methods'] = metadata
    save_config(full_config)

# -------------------------------------------------------

def initialize_data():
    """从 data/fact/ 加载默认轨迹数据到内存，并删除自选平台"""
    # 删除自选平台（如果存在）
    if 'self' in detection_methods:
        del detection_methods['self']
        print("已移除自选平台")

    default_data = load_default_data()
    for mid, data in default_data.items():
        if mid in detection_methods:
            detection_methods[mid]['points'] = data['points']
            detection_methods[mid]['timestamps'] = data['timestamps']
            print(f"📊 {mid}: 加载了 {len(data['points'])} 个轨迹点")
    _save_config_metadata(detection_methods)

initialize_data()


@main_bp.route('/')
def index():
    """主页面：传递检测手段元数据和预测配置"""
    # 读取预测配置，若不存在则使用默认值
    pred_settings = config.get('prediction_settings', {
        'min_points': 1,
        'max_points': 20,
        'default_points': 6
    })
    return render_template('index.html', methods_data=detection_methods, pred_settings=pred_settings)


# ---------------------- 清理所有数据并备份 ----------------------
@main_bp.route('/api/clear_all_data', methods=['POST'])
def clear_all_data():
    """
    清理所有数据：将当前所有检测手段的轨迹备份到 data/backup/ 目录，
    然后清空内存中的轨迹数据，并删除 data/fact 和 data/predict 下的所有文件。
    """
    backup_dir = os.path.join('data', 'backup')
    os.makedirs(backup_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # 备份 fact 目录下的文件
    fact_dir = os.path.join('data', 'fact')
    if os.path.exists(fact_dir):
        for fname in os.listdir(fact_dir):
            src = os.path.join(fact_dir, fname)
            if os.path.isfile(src):
                dst = os.path.join(backup_dir, f"fact_{timestamp}_{fname}")
                shutil.copy2(src, dst)
                print(f"备份 {src} -> {dst}")

    # 备份 predict 目录下的文件
    predict_dir = os.path.join('data', 'predict')
    if os.path.exists(predict_dir):
        for fname in os.listdir(predict_dir):
            src = os.path.join(predict_dir, fname)
            if os.path.isfile(src):
                dst = os.path.join(backup_dir, f"predict_{timestamp}_{fname}")
                shutil.copy2(src, dst)
                print(f"备份 {src} -> {dst}")

    # 备份当前内存中的轨迹（可选，但为了保险）
    for method_id, data in detection_methods.items():
        points = data.get('points', [])
        timestamps = data.get('timestamps', [])
        if not points:
            continue
        backup_file = os.path.join(backup_dir, f"memory_{method_id}_{timestamp}.dat")
        with open(backup_file, 'w', encoding='utf-8') as f:
            for i, p in enumerate(points):
                t = timestamps[i] if i < len(timestamps) else 0
                f.write(f"{p[0]} {p[1]} {p[2]} {t}\n")
        print(f"备份内存 {method_id} 数据至 {backup_file}")

    # 删除 data/fact 下的所有文件
    if os.path.exists(fact_dir):
        for fname in os.listdir(fact_dir):
            file_path = os.path.join(fact_dir, fname)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"删除 {file_path}")

    # 删除 data/predict 下的所有文件
    if os.path.exists(predict_dir):
        for fname in os.listdir(predict_dir):
            file_path = os.path.join(predict_dir, fname)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"删除 {file_path}")

    # 清空内存中的轨迹数据
    for method_id in detection_methods:
        detection_methods[method_id]['points'] = []
        detection_methods[method_id]['timestamps'] = []
    # 保存元数据（不含轨迹）到 config.json
    _save_config_metadata(detection_methods)
    return jsonify({'success': True, 'message': '所有数据已清理并备份'})


# ---------------------- 加载外部数据 ----------------------
@main_bp.route('/api/load_data', methods=['POST'])
def load_data():
    """加载用户上传的 .dat 文件，作为自选检测手段（self）"""
    file = request.files.get('file')
    method_id = request.form.get('method_id')
    if not file or not method_id:
        return jsonify({'success': False, 'error': 'No file or method_id'})

    # 将文件保存到临时路径，再用 load_dat_file 读取（因为 load_dat_file 期望文件路径）
    temp_path = os.path.join('data', 'temp_upload.dat')
    file.save(temp_path)
    points, timestamps = load_dat_file(temp_path)
    os.remove(temp_path)  # 清理临时文件

    if not points:
        return jsonify({'success': False, 'error': 'Invalid file format'})

    if method_id == 'self':
        # 创建自选检测手段（若不存在）
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
        # 只保存元数据到 config
        _save_config_metadata(detection_methods)
        return jsonify({'success': True, 'method_id': method_id, 'name': detection_methods[method_id]['name']})
    else:
        return jsonify({'success': False, 'error': 'Invalid method_id'})


# ---------------------- 预测所有平台 ----------------------
@main_bp.route('/api/predict_all', methods=['POST'])
def predict_all():
    """对当前所有可见检测手段进行预测，并将结果保存到 data/predict/，同时返回预测点和对应时间戳"""
    data = request.json or {}
    num_points = data.get('num_points', 6)
    time_step = data.get('time_step', 0.5)

    # 从配置读取范围限制
    pred_settings = config.get('prediction_settings', {'min_points': 1, 'max_points': 20})
    min_pts = pred_settings.get('min_points', 1)
    max_pts = pred_settings.get('max_points', 20)
    num_points = max(min_pts, min(num_points, max_pts))  # 限制在配置范围内

    results = {}
    for method_id, method_data in detection_methods.items():
        if method_data.get('visible', False) and method_data.get('points') and len(method_data['points']) >= 2:
            points = method_data['points']
            timestamps = method_data.get('timestamps', [])
            pred_points, pred_times = generate_prediction(points, timestamps, num_points, time_step)
            if pred_points:
                save_predict_data(method_id, pred_points, pred_times)
                results[method_id] = {'prediction': pred_points, 'pred_times': pred_times}
    return jsonify({'success': True, 'results': results})


# ---------------------- 单平台预测 ----------------------
@main_bp.route('/api/predict', methods=['POST'])
def predict():
    """
    对指定平台进行预测，返回预测点及对应时间戳
    """
    data = request.json
    method_id = data.get('method_id')
    points = data.get('points', [])
    timestamps = data.get('timestamps', [])
    num_points = data.get('num_points', 6)
    time_step = data.get('time_step', 0.5)

    # 从配置读取范围限制
    pred_settings = config.get('prediction_settings', {'min_points': 1, 'max_points': 20})
    min_pts = pred_settings.get('min_points', 1)
    max_pts = pred_settings.get('max_points', 20)
    num_points = max(min_pts, min(num_points, max_pts))  # 限制在配置范围内

    pred_points, pred_times = generate_prediction(points, timestamps, num_points, time_step)
    if pred_points:
        save_predict_data(method_id, pred_points, pred_times)
    return jsonify({'success': True, 'prediction': pred_points, 'pred_times': pred_times})


# ---------------------- 刷新数据（重新加载默认 fact 文件）---------------------
@main_bp.route('/api/refresh_data', methods=['POST'])
def refresh_data():
    """重新加载默认 data/fact/*.dat 文件，只更新 visible/infrared/radar，保留 self 数据"""
    default_data = load_default_data()
    for mid, data in default_data.items():
        if mid in detection_methods:
            detection_methods[mid]['points'] = data['points']
            detection_methods[mid]['timestamps'] = data['timestamps']
    # 保留 self 数据，不做任何操作
    _save_config_metadata(detection_methods)
    return jsonify({'success': True})


# ---------------------- AI 建议 ----------------------
@main_bp.route('/api/ai_suggestion', methods=['POST'])
def ai_suggestion():
    data = request.json
    methods_data = data.get('methods_data', {})
    try:
        suggestion = get_ai_suggestion(methods_data, SF_API_KEY, SF_URL, SF_MODEL)
        return jsonify({'success': True, 'suggestion': suggestion})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ---------------------- 备份管理 ----------------------
@main_bp.route('/api/list_backups', methods=['GET'])
def list_backups():
    """列出 data/backup/ 目录下的所有备份文件"""
    backup_dir = os.path.join('data', 'backup')
    if not os.path.exists(backup_dir):
        return jsonify({'success': True, 'backups': []})
    files = []
    for f in os.listdir(backup_dir):
        if f.endswith('.dat'):
            # 解析文件名，提取平台和时间戳
            parts = f.split('_')
            if len(parts) >= 2:
                method = parts[0]
                timestamp = parts[1].replace('.dat', '')
                files.append({
                    'filename': f,
                    'method': method,
                    'timestamp': timestamp,
                    'full_path': os.path.join(backup_dir, f)
                })
    files.sort(key=lambda x: x['timestamp'], reverse=True)  # 最新在前
    return jsonify({'success': True, 'backups': files})


@main_bp.route('/api/restore_backup', methods=['POST'])
def restore_backup():
    """从选定的备份文件恢复数据（可恢复单个平台或全部平台）"""
    data = request.json
    backup_file = data.get('backup_file')  # 完整文件名
    if not backup_file:
        return jsonify({'success': False, 'error': 'No backup file specified'})

    backup_path = os.path.join('data', 'backup', backup_file)
    if not os.path.exists(backup_path):
        return jsonify({'success': False, 'error': 'Backup file not found'})

    # 加载备份文件中的轨迹
    points, timestamps = load_dat_file(backup_path)
    if not points:
        return jsonify({'success': False, 'error': 'Invalid backup file'})

    # 根据文件名判断是哪个平台
    method_id = backup_file.split('_')[0]
    if method_id not in detection_methods:
        return jsonify({'success': False, 'error': f'Unknown method: {method_id}'})

    # 恢复数据到该平台
    detection_methods[method_id]['points'] = points
    detection_methods[method_id]['timestamps'] = timestamps
    _save_config_metadata(detection_methods)

    return jsonify({'success': True, 'message': f'已从 {backup_file} 恢复 {method_id} 数据'})


@main_bp.route('/api/restore_all_backups', methods=['POST'])
def restore_all_backups():
    """恢复所有备份（将最新备份文件恢复到对应平台）"""
    backup_dir = os.path.join('data', 'backup')
    if not os.path.exists(backup_dir):
        return jsonify({'success': False, 'error': 'No backup directory'})

    # 收集每个平台的最新备份
    latest_backups = {}
    for f in os.listdir(backup_dir):
        if not f.endswith('.dat'):
            continue
        parts = f.split('_')
        if len(parts) >= 2:
            method = parts[0]
            timestamp = parts[1].replace('.dat', '')
            if method not in latest_backups or timestamp > latest_backups[method]['timestamp']:
                latest_backups[method] = {'filename': f, 'timestamp': timestamp}

    restored = []
    for method_id, info in latest_backups.items():
        if method_id not in detection_methods:
            continue
        backup_path = os.path.join(backup_dir, info['filename'])
        points, timestamps = load_dat_file(backup_path)
        if points:
            detection_methods[method_id]['points'] = points
            detection_methods[method_id]['timestamps'] = timestamps
            restored.append(method_id)

    if restored:
        _save_config_metadata(detection_methods)
        return jsonify({'success': True, 'message': f'已恢复 {len(restored)} 个平台', 'restored': restored})
    else:
        return jsonify({'success': False, 'error': '没有找到可恢复的备份'})