"""
数据与备份 API — 轨迹管理 + 备份恢复

路由表:
  POST /api/refresh_data       刷新默认轨迹
  POST /api/load_data          上传自选数据
  POST /api/clear_all_data     清理并自动备份
  GET  /api/list_backups        列出备份快照
  POST /api/restore_backup      恢复指定快照
  POST /api/restore_all_backups 恢复最新快照
  POST /api/backup/create       手动创建备份
  POST /api/backup/delete       删除备份
"""
import os
from flask import Blueprint, jsonify, request

from trajectory_reconstruction.services import data_service, backup_service
from trajectory_reconstruction.services.data_service import save_metadata
from trajectory_reconstruction.core.state import detection_methods
from trajectory_reconstruction.core.io.data_loader import load_dat_file

api_bp = Blueprint('api', __name__)

# ═══════════════════════════ 数据管理 ═══════════════════════════

@api_bp.route('/api/refresh_data', methods=['POST'])
def refresh_data():
    data_service.refresh_fact_data()
    return jsonify({'success': True})


@api_bp.route('/api/load_data', methods=['POST'])
def load_data():
    file = request.files.get('file')
    method_id = request.form.get('method_id', '')

    if not file or method_id != 'self':
        return jsonify({'success': False, 'error': '参数错误'}), 400

    temp = os.path.join('data', 'temp_upload.dat')
    file.save(temp)
    points, timestamps = load_dat_file(temp)
    os.remove(temp)

    if not points:
        return jsonify({'success': False, 'error': '文件格式无效（需要 x y z t）'}), 400

    info = data_service.load_self_data(points, timestamps)
    return jsonify({'success': True, 'method_id': 'self', 'name': info['name']})


@api_bp.route('/api/clear_all_data', methods=['POST'])
def clear_all_data():
    name = data_service.clear_all_data()
    return jsonify({'success': True, 'message': f'数据已清理，备份: {name}', 'backup_name': name})

# ═══════════════════════════ 备份管理 ═══════════════════════════

@api_bp.route('/api/list_backups', methods=['GET'])
def list_backups():
    backups = backup_service.list_backups()
    return jsonify({'success': True, 'backups': backups})


@api_bp.route('/api/restore_backup', methods=['POST'])
def restore_backup():
    data = request.get_json(silent=True) or {}
    backup_file = data.get('backup_file', '')
    if not backup_file:
        return jsonify({'success': False, 'error': '缺少 backup_file'}), 400

    ok, msg = backup_service.restore_backup(backup_file)
    return jsonify({'success': ok, 'message' if ok else 'error': msg}), (200 if ok else 400)


@api_bp.route('/api/restore_all_backups', methods=['POST'])
def restore_all_backups():
    restored, msg = backup_service.restore_all_latest()
    if restored:
        return jsonify({'success': True, 'message': msg, 'restored': restored})
    return jsonify({'success': False, 'error': msg}), 400


@api_bp.route('/api/backup/create', methods=['POST'])
def create_backup():
    """手动创建备份快照"""
    name = backup_service.create_backup(label='manual')
    return jsonify({'success': True, 'message': f'备份已创建: {name}', 'name': name})


@api_bp.route('/api/backup/delete', methods=['POST'])
def delete_backup():
    """删除指定备份快照"""
    data = request.get_json(silent=True) or {}
    name = data.get('backup_name', '')
    if not name:
        return jsonify({'success': False, 'error': '缺少 backup_name'}), 400
    ok, msg = backup_service.delete_backup(name)
    return jsonify({'success': ok, 'message' if ok else 'error': msg}), (200 if ok else 400)


@api_bp.route('/api/toggle_method', methods=['POST'])
def toggle_method():
    """切换检测平台可见性（禁用平台不可切换）"""
    data = request.get_json(silent=True) or {}
    method_id = data.get('method_id', '')
    if not method_id or method_id not in detection_methods:
        return jsonify({'success': False, 'error': '未知平台'}), 400
    if not detection_methods[method_id].get('enabled', True):
        return jsonify({'success': False, 'error': '平台已禁用'}), 400
    detection_methods[method_id]['visible'] = not detection_methods[method_id].get('visible', True)
    save_metadata()
    return jsonify({'success': True, 'visible': detection_methods[method_id]['visible']})


@api_bp.route('/api/synthesize', methods=['POST'])
def synthesize():
    """加权合成综合轨迹"""
    result = data_service.synthesize_trajectory()
    code = 200 if result.get('success') else 400
    return jsonify(result), code
