"""
备份服务 — 快照目录式备份管理

快照结构:
  data/backup/<YYYYmmdd_HHMMSS>_<label>/
    ├── manifest.json          # 元信息
    ├── fact/                  # 复制自 data/fact/
    ├── predict/               # 复制自 data/predict/
    └── memory/                # 内存轨迹 dump
"""
import json
import os
import shutil
import time

from trajectory_reconstruction.core.state import detection_methods
from trajectory_reconstruction.core.config.config_manager import PROJECT_ROOT


BACKUP_DIR = os.path.join(PROJECT_ROOT, 'data', 'backup')
FACT_DIR = os.path.join(PROJECT_ROOT, 'data', 'fact')
PREDICT_DIR = os.path.join(PROJECT_ROOT, 'data', 'predict')

# ---- 内部辅助 ----

def _snapshot_path(name: str) -> str:
    return os.path.join(BACKUP_DIR, name)


def _read_manifest(name: str) -> dict | None:
    """读取快照的 manifest.json，不存在或损坏返回 None"""
    path = os.path.join(_snapshot_path(name), 'manifest.json')
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def _write_manifest(name: str, label: str) -> dict:
    """根据当前状态生成并写入 manifest.json"""
    created = time.strftime('%Y-%m-%d %H:%M:%S')
    methods_info = {}
    for mid, data in detection_methods.items():
        if not data.get('points'):
            continue  # 跳过空平台
        methods_info[mid] = {
            'name': data.get('name', ''),
            'color': data.get('color', ''),
            'point_count': len(data.get('points', [])),
        }

    snap_path = _snapshot_path(name)
    fact_files = os.listdir(os.path.join(snap_path, 'fact')) if os.path.isdir(os.path.join(snap_path, 'fact')) else []
    predict_files = os.listdir(os.path.join(snap_path, 'predict')) if os.path.isdir(os.path.join(snap_path, 'predict')) else []
    memory_files = os.listdir(os.path.join(snap_path, 'memory')) if os.path.isdir(os.path.join(snap_path, 'memory')) else []

    manifest = {
        'created': created,
        'label': label,
        'methods': methods_info,
        'files': {
            'fact': fact_files,
            'predict': predict_files,
            'memory': memory_files,
        },
    }
    with open(os.path.join(snap_path, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest


def _copy_dir(src: str, dst: str):
    """复制目录下所有常规文件到目标目录"""
    if not os.path.isdir(src):
        return
    os.makedirs(dst, exist_ok=True)
    for fname in os.listdir(src):
        fp = os.path.join(src, fname)
        if os.path.isfile(fp):
            shutil.copy2(fp, os.path.join(dst, fname))


# ---- 公共 API ----

def create_backup(label: str = 'manual') -> str:
    """
    创建快照备份。
    返回快照目录名。
    """
    # 自动标签：只用实际有磁盘文件的平台名拼接
    if label in ('manual', 'auto'):
        active_platforms = []
        if os.path.isdir(FACT_DIR):
            for fname in sorted(os.listdir(FACT_DIR)):
                if not fname.endswith(('.npz', '.dat')):
                    continue
                for mid, data in detection_methods.items():
                    # 匹配 .npz 或 .dat
                    matched = False
                    for ext in ('.npz', '.dat'):
                        if fname == f'{mid}{ext}':
                            matched = True; break
                    if matched and data.get('points'):
                        active_platforms.append(data.get('name', mid))
        if active_platforms:
            label = '+'.join(active_platforms)
    name = f"{time.strftime('%Y%m%d_%H%M%S')}_{label}"
    snap_path = _snapshot_path(name)

    os.makedirs(os.path.join(snap_path, 'fact'), exist_ok=True)
    os.makedirs(os.path.join(snap_path, 'predict'), exist_ok=True)
    os.makedirs(os.path.join(snap_path, 'memory'), exist_ok=True)

    # 复制磁盘数据
    _copy_dir(FACT_DIR, os.path.join(snap_path, 'fact'))
    _copy_dir(PREDICT_DIR, os.path.join(snap_path, 'predict'))

    # Dump 内存数据
    for mid, data in detection_methods.items():
        pts = data.get('points', [])
        tss = data.get('timestamps', [])
        if not pts:
            continue
        mem_path = os.path.join(snap_path, 'memory', f'{mid}.npz')
        import numpy as np
        np.savez(mem_path,
                 positions=np.array(pts, dtype=np.float32),
                 timestamps=np.array(tss, dtype=np.float32))

    _write_manifest(name, label)
    print(f'[OK] 备份已创建: {name}')
    return name


def list_backups() -> list[dict]:
    """
    列出所有快照（按时间倒序）。
    返回格式兼容旧前端 key。
    """
    if not os.path.isdir(BACKUP_DIR):
        return []

    result = []
    for entry in sorted(os.listdir(BACKUP_DIR), reverse=True):
        snap_path = os.path.join(BACKUP_DIR, entry)
        if not os.path.isdir(snap_path):
            continue
        manifest = _read_manifest(entry)
        if manifest is None:
            continue

        methods = manifest.get('methods', {})
        method_names = [m['name'] for m in methods.values()] if methods else []
        result.append({
            'filename': entry,
            'name': entry,
            'timestamp': manifest.get('created', ''),
            'created': manifest.get('created', ''),
            'method': ' + '.join(method_names) if method_names else '空',
            'method_summary': ' + '.join(method_names) if method_names else '空',
            'label': manifest.get('label') or 'manual',
            'methods': methods,
            'point_count': sum(m.get('point_count', 0) for m in methods.values()),
        })

    return result


def _clear_all():
    """清空所有数据和文件（为恢复做准备）"""
    # 清空内存
    for mid in detection_methods:
        detection_methods[mid]['points'] = []
        detection_methods[mid]['timestamps'] = []
    # 删除自选和综合平台（恢复后会重新合成）
    if 'self' in detection_methods:
        del detection_methods['self']
    if 'synthetic' in detection_methods:
        del detection_methods['synthetic']
    # 清空磁盘文件
    for d in (FACT_DIR, PREDICT_DIR):
        if os.path.isdir(d):
            for fname in os.listdir(d):
                fp = os.path.join(d, fname)
                if os.path.isfile(fp):
                    os.remove(fp)


def restore_backup(name: str, full: bool = False) -> tuple[bool, str]:
    """
    恢复备份：先备份当前数据，再释放备份中的 fact/predict 文件。
    full=True: 清空全部后完整恢复
    full=False: 部分恢复（只覆盖备份中存在的文件）
    返回 (success, message)。
    """
    snap_path = _snapshot_path(name)
    if not os.path.isdir(snap_path):
        return False, f'备份不存在: {name}'

    manifest = _read_manifest(name)
    if manifest is None:
        return False, '备份 manifest 丢失或损坏'

    from trajectory_reconstruction.core.io.data_loader import load_trajectory_file
    from trajectory_reconstruction.services.data_service import save_metadata, refresh_fact_data

    restored_count = 0

    # 完整恢复：先清空
    if full:
        _clear_all()
        os.makedirs(FACT_DIR, exist_ok=True)
        os.makedirs(PREDICT_DIR, exist_ok=True)

    # 释放 fact/ 文件
    fact_src = os.path.join(snap_path, 'fact')
    fact_dst = FACT_DIR
    if os.path.isdir(fact_src):
        os.makedirs(fact_dst, exist_ok=True)
        for fname in os.listdir(fact_src):
            src_fp = os.path.join(fact_src, fname)
            if os.path.isfile(src_fp):
                shutil.copy2(src_fp, os.path.join(fact_dst, fname))
                restored_count += 1

    # 释放 predict/ 文件
    predict_src = os.path.join(snap_path, 'predict')
    predict_dst = PREDICT_DIR
    if os.path.isdir(predict_src):
        os.makedirs(predict_dst, exist_ok=True)
        for fname in os.listdir(predict_src):
            src_fp = os.path.join(predict_src, fname)
            if os.path.isfile(src_fp):
                shutil.copy2(src_fp, os.path.join(predict_dst, fname))
                restored_count += 1

    # 从磁盘重新加载 fact 文件到内存，重新合成（保留 self 数据）
    refresh_fact_data(keep_self=True)

    mode = '完整' if full else '部分'
    return True, f'已从 {name} {mode}恢复 {restored_count} 个文件'


def restore_all_latest() -> tuple[list[str], str]:
    """一键恢复最新快照（完整覆盖）"""
    backups = list_backups()
    if not backups:
        return [], '没有可用的备份'
    latest = backups[0]['name']
    ok, msg = restore_backup(latest, full=True)
    if ok:
        return [latest], msg
    return [], msg


def delete_backup(name: str) -> tuple[bool, str]:
    """删除指定快照"""
    snap_path = _snapshot_path(name)
    if not os.path.isdir(snap_path):
        return False, f'备份不存在: {name}'
    shutil.rmtree(snap_path)
    return True, f'已删除备份 {name}'


def migrate_legacy() -> int:
    """
    清理旧版备份文件（扁平 .dat 文件和 fact/predict 子目录）。
    保留合法的快照目录（含有 manifest.json）。
    返回清理的文件数。
    """
    if not os.path.isdir(BACKUP_DIR):
        return 0

    removed = 0
    for entry in os.listdir(BACKUP_DIR):
        full = os.path.join(BACKUP_DIR, entry)

        # 删除扁平轨迹文件
        if os.path.isfile(full) and entry.endswith(('.npz', '.dat')):
            os.remove(full)
            removed += 1

        # 删除旧版 fact/ predict/ 子目录（不含 manifest.json）
        elif os.path.isdir(full) and entry in ('fact', 'predict'):
            shutil.rmtree(full)
            removed += 1

    if removed:
        print(f'[OK] 清理了 {removed} 个旧备份文件/目录')
    return removed
