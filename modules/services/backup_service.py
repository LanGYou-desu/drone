"""
备份/恢复服务 — 管理 data/backup/ 目录下的备份文件
"""
import os
from typing import Optional

from modules.state import detection_methods
from modules.services.data_service import save_metadata
from modules.data.data_loader import load_dat_file


BACKUP_DIR = os.path.join('data', 'backup')


def list_backups() -> list[dict]:
    """
    列出所有备份文件
    返回 [ { filename, method, timestamp, full_path }, ... ]（按时间倒序）
    """
    if not os.path.exists(BACKUP_DIR):
        return []

    files = []
    for fname in os.listdir(BACKUP_DIR):
        if not fname.endswith('.dat'):
            continue
        parts = fname.split('_')
        if len(parts) >= 2:
            method = parts[0]
            timestamp = parts[1].replace('.dat', '')
            files.append({
                'filename': fname,
                'method': method,
                'timestamp': timestamp,
                'full_path': os.path.join(BACKUP_DIR, fname),
            })

    files.sort(key=lambda x: x['timestamp'], reverse=True)
    return files


def restore_backup(backup_file: str) -> tuple[bool, str]:
    """
    从指定备份文件恢复数据到对应平台
    返回 (success, message)
    """
    backup_path = os.path.join(BACKUP_DIR, backup_file)
    if not os.path.exists(backup_path):
        return False, '备份文件不存在'

    points, timestamps = load_dat_file(backup_path)
    if not points:
        return False, '备份文件格式无效'

    method_id = backup_file.split('_')[0]
    if method_id not in detection_methods:
        return False, f'未知平台: {method_id}'

    detection_methods[method_id]['points'] = points
    detection_methods[method_id]['timestamps'] = timestamps
    save_metadata()
    return True, f'已从 {backup_file} 恢复 {method_id} 数据'


def restore_all_latest() -> tuple[list[str], str]:
    """
    一键恢复所有平台的最新备份
    返回 (restored_method_ids, message)
    """
    if not os.path.exists(BACKUP_DIR):
        return [], '备份目录不存在'

    # 收集每个平台的最新备份
    latest: dict[str, dict] = {}
    for fname in os.listdir(BACKUP_DIR):
        if not fname.endswith('.dat'):
            continue
        parts = fname.split('_')
        if len(parts) < 2:
            continue
        method = parts[0]
        ts = parts[1].replace('.dat', '')
        if method not in latest or ts > latest[method]['timestamp']:
            latest[method] = {'filename': fname, 'timestamp': ts}

    restored = []
    for mid, info in latest.items():
        if mid not in detection_methods:
            continue
        backup_path = os.path.join(BACKUP_DIR, info['filename'])
        points, timestamps = load_dat_file(backup_path)
        if points:
            detection_methods[mid]['points'] = points
            detection_methods[mid]['timestamps'] = timestamps
            restored.append(mid)

    if restored:
        save_metadata()
        return restored, f'已恢复 {len(restored)} 个平台'
    return [], '没有找到可恢复的备份'
