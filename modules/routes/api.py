"""
数据与备份 API — 轨迹管理 + 备份恢复

路由表:
  POST /api/refresh_data       刷新默认轨迹
  POST /api/load_data          上传自选数据
  POST /api/clear_all_data     清理并备份
  GET  /api/list_backups        列出备份
  POST /api/restore_backup      恢复指定备份
  POST /api/restore_all_backups 一键恢复
"""
import os
from flask import Blueprint, jsonify, request

from modules.services import data_service, backup_service
from modules.data.data_loader import load_dat_file

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
    data_service.clear_all_data()
    return jsonify({'success': True, 'message': '数据已清理并备份'})

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
