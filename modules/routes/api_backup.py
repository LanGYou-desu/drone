"""
备份管理 API — 列出、恢复备份
"""
from flask import Blueprint, jsonify, request

from modules.services import backup_service

backup_bp = Blueprint('backup', __name__)


@backup_bp.route('/api/list_backups', methods=['GET'])
def list_backups():
    """列出所有备份文件"""
    backups = backup_service.list_backups()
    return jsonify({'success': True, 'backups': backups})


@backup_bp.route('/api/restore_backup', methods=['POST'])
def restore_backup():
    """
    从指定备份恢复数据
    Body: { backup_file: "visible_20260627_143052.dat" }
    """
    data = request.get_json(silent=True) or {}
    backup_file = data.get('backup_file', '')

    if not backup_file:
        return jsonify({'success': False, 'error': '缺少 backup_file'}), 400

    ok, msg = backup_service.restore_backup(backup_file)
    if ok:
        return jsonify({'success': True, 'message': msg})
    return jsonify({'success': False, 'error': msg}), 400


@backup_bp.route('/api/restore_all_backups', methods=['POST'])
def restore_all_backups():
    """一键恢复所有平台最新备份"""
    restored, msg = backup_service.restore_all_latest()
    if restored:
        return jsonify({'success': True, 'message': msg, 'restored': restored})
    return jsonify({'success': False, 'error': msg}), 400
