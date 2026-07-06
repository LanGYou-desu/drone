"""
检测 API — Detection API

双目无人机检测的 HTTP 端点实现。
"""

import json
import os

from flask import jsonify, request, send_file

from trajectory_recognition.views import api_detection_bp
from trajectory_recognition.services.detection_service import (
    create_session, start_detection, stop_detection,
    pause_detection, resume_detection,
    get_session, get_active_session, list_sessions, delete_session,
)
from trajectory_recognition.services.data_bridge import (
    tracks_to_dat, backup_existing_fact, list_detect_files,
)


def _cleanup_uploads(upload_dir: str = None):
    """清理临时上传文件"""
    import glob
    d = upload_dir or os.path.join(os.getcwd(), 'data', 'uploads')
    if os.path.isdir(d):
        for f in glob.glob(os.path.join(d, 'temp_*')):
            try:
                os.remove(f)
            except Exception:
                pass


# ── 启动 / 停止 / 暂停 ──────────────────────────

@api_detection_bp.route('/start', methods=['POST'])
def start():
    """启动双目检测"""
    try:
        # 支持 FormData（文件上传）和 JSON（流地址）
        if request.files:
            # 文件上传模式
            file_a = request.files.get('video_a')
            file_b = request.files.get('video_b')
            config = json.loads(request.form.get('config', '{}'))

            if not file_a or not file_b:
                return jsonify({'success': False, 'error': '需要两个视频文件（video_a + video_b）'}), 400

            # 清理旧上传 + 保存临时文件
            upload_dir = os.path.join(os.getcwd(), 'data', 'uploads')
            _cleanup_uploads(upload_dir)
            os.makedirs(upload_dir, exist_ok=True)
            path_a = os.path.join(upload_dir, f"temp_a_{file_a.filename}")
            path_b = os.path.join(upload_dir, f"temp_b_{file_b.filename}")
            file_a.save(path_a)
            file_b.save(path_b)

            source_a, source_b = path_a, path_b
        else:
            data = request.get_json(silent=True) or {}
            source_a = data.get('source_a')
            source_b = data.get('source_b')
            config = data

            if not source_a or not source_b:
                return jsonify({'success': False, 'error': '缺少 source_a 或 source_b'}), 400

        platform_id = config.get('platform_id', 'visible')
        session = create_session(source_a, source_b, platform_id, config)
        start_detection(session.session_id)

        return jsonify({
            'success': True,
            'session': session.to_dict(),
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@api_detection_bp.route('/stop', methods=['POST'])
def stop():
    data = request.get_json(silent=True) or {}
    sid = data.get('session_id') or (get_active_session().session_id if get_active_session() else None)
    if not sid:
        return jsonify({'success': False, 'error': '没有活跃的检测会话'}), 400
    stop_detection(sid)
    return jsonify({'success': True})


@api_detection_bp.route('/pause', methods=['POST'])
def pause():
    data = request.get_json(silent=True) or {}
    sid = data.get('session_id') or (get_active_session().session_id if get_active_session() else None)
    if not sid:
        return jsonify({'success': False, 'error': '没有活跃的检测会话'}), 400
    pause_detection(sid)
    return jsonify({'success': True})


@api_detection_bp.route('/resume', methods=['POST'])
def resume():
    data = request.get_json(silent=True) or {}
    sid = data.get('session_id') or (get_active_session().session_id if get_active_session() else None)
    if not sid:
        return jsonify({'success': False, 'error': '没有可恢复的会话'}), 400
    resume_detection(sid)
    return jsonify({'success': True})


# ── 状态 / 跟踪 / 预览 ──────────────────────────

@api_detection_bp.route('/status', methods=['GET'])
def status():
    sid = request.args.get('session_id')
    session = get_session(sid) if sid else get_active_session()
    if not session:
        return jsonify({'success': True, 'session': None})
    return jsonify({'success': True, 'session': session.to_dict()})


@api_detection_bp.route('/tracks', methods=['GET'])
def tracks():
    sid = request.args.get('session_id')
    session = get_session(sid) if sid else get_active_session()
    if not session or not session._tracker:
        return jsonify({'success': True, 'tracks': []})

    fmt = request.args.get('format', 'summary')
    all_tracks = session._tracker.get_all_tracks()

    if fmt == 'full':
        result = []
        for t in all_tracks:
            d = t.to_summary()
            d['positions'] = t.positions
            d['timestamps'] = t.timestamps
            d['bboxes'] = t.bboxes
            result.append(d)
        return jsonify({'success': True, 'tracks': result})

    return jsonify({
        'success': True,
        'tracks': [t.to_summary() for t in all_tracks],
    })


@api_detection_bp.route('/preview', methods=['GET'])
def preview():
    """返回最新预览帧 JPEG"""
    channel = request.args.get('channel', 'a')  # 'a' 左目 / 'b' 右目
    sid = request.args.get('session_id')
    session = get_session(sid) if sid else get_active_session()

    if not session:
        return '', 204  # 无会话，前端静默跳过
    frame = session._frame_a if channel == 'a' else session._frame_b
    if not frame:
        return '', 204  # 暂无帧，等待检测产出

    from flask import Response
    return Response(frame, mimetype='image/jpeg',
                    headers={'Cache-Control': 'no-cache'})


# ── 备份列表（自给自足，不依赖 :5000）──

@api_detection_bp.route('/backups', methods=['GET'])
def list_backups():
    """列出 data/backup/ 中的所有备份（统一 manifest 格式）"""
    try:
        import json as _json
        backup_dir = os.path.join(os.getcwd(), 'data', 'backup')
        if not os.path.isdir(backup_dir):
            return jsonify({'success': True, 'backups': []})

        backups = []
        for name in sorted(os.listdir(backup_dir), reverse=True):
            d = os.path.join(backup_dir, name)
            if not os.path.isdir(d):
                continue
            manifest_path = os.path.join(d, 'manifest.json')
            manifest = {}
            if os.path.isfile(manifest_path):
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    manifest = _json.load(f)

            # 统一格式: {created, label, files: {fact:[], predict:[], memory:[]}}
            files_dict = manifest.get('files', {})
            fact_files = files_dict.get('fact', []) if isinstance(files_dict, dict) else []
            pt_count = 0
            for ff in fact_files:
                fp = os.path.join(d, 'fact', ff)
                if os.path.isfile(fp):
                    pt_count += sum(1 for _ in open(fp, 'r'))

            backups.append({
                'name': name,
                'timestamp': manifest.get('created', ''),
                'label': manifest.get('label') or 'auto',
                'file_count': len(fact_files),
                'total_points': pt_count,
            })
        return jsonify({'success': True, 'backups': backups})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ── 文件管理 ──────────────────────────────────

@api_detection_bp.route('/files', methods=['GET'])
def list_files():
    try:
        files = list_detect_files()
        return jsonify({
            'success': True,
            'files': files,
            'total': len(files),
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@api_detection_bp.route('/files/<filename>', methods=['DELETE'])
def delete_file(filename: str):
    try:
        # 安全检查：只允许删除 .dat 文件
        if not filename.endswith('.dat') or '..' in filename or '/' in filename:
            return jsonify({'success': False, 'error': '无效的文件名'}), 400

        fpath = os.path.join(os.getcwd(), 'data', 'fact', filename)
        if os.path.isfile(fpath):
            os.remove(fpath)
            return jsonify({'success': True, 'message': f'已删除 {filename}'})
        return jsonify({'success': False, 'error': '文件不存在'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ── 配置 ─────────────────────────────────────

@api_detection_bp.route('/config', methods=['POST'])
def save_config():
    try:
        data = request.get_json(silent=True) or {}
        config_path = os.path.join(os.getcwd(), 'config.json')

        current = {}
        if os.path.isfile(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                current = json.load(f)

        # 合并 detection 配置
        if 'detection' not in current:
            current['detection'] = {}
        det_keys = ['model', 'device', 'input_width', 'input_height',
                     'confidence_threshold', 'nms_threshold',
                     'frame_interval', 'tracker', 'auto_save', 'target_classes']
        for k in det_keys:
            if k in data:
                current['detection'][k] = data[k]

        # 合并 platforms 配置
        if 'platforms' in data:
            if 'platforms' not in current:
                current['platforms'] = {}
            current['platforms'].update(data['platforms'])

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(current, f, ensure_ascii=False, indent=4)

        return jsonify({'success': True, 'config': current})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@api_detection_bp.route('/config', methods=['GET'])
def get_config():
    try:
        config_path = os.path.join(os.getcwd(), 'config.json')
        if os.path.isfile(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return jsonify({'success': True, 'config': json.load(f)})
        return jsonify({'success': True, 'config': {}})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ── 保存结果 ─────────────────────────────────

@api_detection_bp.route('/save', methods=['POST'])
def save_results():
    """手动保存检测结果到 data/fact/"""
    try:
        data = request.get_json(silent=True) or {}
        sid = data.get('session_id')
        session = get_session(sid) if sid else get_active_session()

        if not session:
            return jsonify({'success': False, 'error': '没有可用的检测会话'}), 400

        platform_id = data.get('platform_id') or session.platform_id
        tracker = session._tracker

        if not tracker:
            return jsonify({'success': False, 'error': '没有跟踪数据可保存'}), 400

        tracks = tracker.get_all_tracks()

        # 确定目标文件
        from trajectory_recognition.services.data_bridge import PLATFORM_FACT_MAP
        target_file = PLATFORM_FACT_MAP.get(platform_id, f"{platform_id}.dat")

        # 手动保存先备份旧数据再写入
        backup_path = backup_existing_fact(filenames=[target_file], label='manual')
        files = tracks_to_dat(tracks, platform_id=platform_id, auto_backup=False)

        # 通知重建模块刷新
        from trajectory_recognition.services.data_bridge import notify_reconstruction
        notify_reconstruction()

        return jsonify({
            'success': True,
            'files': files,
            'track_count': len(files),
            'platform': platform_id,
            'backup': backup_path,
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
