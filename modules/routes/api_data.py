"""
数据管理 API — 刷新、上传、清理轨迹数据
"""
import os
from flask import Blueprint, jsonify, request

from modules.services import data_service
from modules.services.state import detection_methods
from modules.data.data_loader import load_dat_file

data_bp = Blueprint('data', __name__)


@data_bp.route('/api/refresh_data', methods=['POST'])
def refresh_data():
    """重新加载默认 fact 数据"""
    updated = data_service.refresh_fact_data()
    return jsonify({'success': True, 'updated': updated})


@data_bp.route('/api/load_data', methods=['POST'])
def load_data():
    """
    上传 .dat 文件作为自选平台（self）轨迹
    FormData: file, method_id='self'
    """
    file = request.files.get('file')
    method_id = request.form.get('method_id')
    if not file or not method_id:
        return jsonify({'success': False, 'error': '缺少文件或 method_id'}), 400
    if method_id != 'self':
        return jsonify({'success': False, 'error': '仅支持 method_id=self'}), 400

    # 保存到临时文件
    temp_path = os.path.join('data', 'temp_upload.dat')
    file.save(temp_path)
    points, timestamps = load_dat_file(temp_path)
    os.remove(temp_path)

    if not points:
        return jsonify({'success': False, 'error': '文件格式无效：需要每行 x y z t'}), 400

    info = data_service.load_self_data(points, timestamps)
    return jsonify({
        'success': True,
        'method_id': method_id,
        'name': info['name'],
    })


@data_bp.route('/api/clear_all_data', methods=['POST'])
def clear_all_data():
    """清理所有数据并备份"""
    backup_dir = data_service.clear_all_data()
    return jsonify({
        'success': True,
        'message': '所有数据已清理并备份',
        'backup_dir': backup_dir,
    })
