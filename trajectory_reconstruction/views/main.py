"""
页面路由 — 多页面渲染入口
每个页面独立渲染，通过模板注入后端数据
"""
import os

from flask import Blueprint, render_template, request, jsonify

from trajectory_reconstruction.core.state import detection_methods
from trajectory_reconstruction.core.config.config_manager import ensure_config, save_config

main_bp = Blueprint('main', __name__)


_METHOD_ORDER = ['visible', 'infrared', 'radar', 'self', 'synthetic']

def _page_context(active: str) -> dict:
    """构建页面公共上下文"""
    cfg = ensure_config()
    self_exists = os.path.isfile(os.path.join('data', 'fact', 'self.dat'))
    ordered = {}
    for mid in _METHOD_ORDER:
        if mid in detection_methods:
            if mid == 'self' and not self_exists:
                continue
            ordered[mid] = detection_methods[mid]
    for mid in detection_methods:
        if mid not in ordered:
            if mid == 'self' and not self_exists:
                continue
            ordered[mid] = detection_methods[mid]
    return {
        'methods_data': ordered,
        'pred_settings': cfg.get('prediction_settings', {}),
        'active_page': active,
        'theme': cfg.get('theme', 'dark'),
        'camera_speed': cfg.get('camera_speed', 0.12),
        'capture_weights': cfg.get('capture_weights', {}),
    }


@main_bp.route('/')
def dashboard():
    """总览页 — 全屏 3D 轨迹视图"""
    ctx = _page_context('dashboard')
    return render_template('index.html', **ctx)


@main_bp.route('/predict')
def predict_page():
    """预测页 — 轨迹预测 + 3D 预览"""
    ctx = _page_context('predict')
    return render_template('predict.html', **ctx)


@main_bp.route('/ai')
def ai_page():
    """AI 策略页 — 大模型捕捉建议"""
    ctx = _page_context('ai')
    return render_template('ai.html', **ctx)


@main_bp.route('/data')
def data_page():
    """数据管理页 — 上传/备份/恢复"""
    ctx = _page_context('data')
    return render_template('data.html', **ctx)


@main_bp.route('/docs')
def docs_page():
    """帮助文档中心"""
    ctx = _page_context('docs')
    return render_template('docs.html', **ctx)


@main_bp.route('/settings')
def settings_page():
    """设置页面 — 主题切换等"""
    ctx = _page_context('settings')
    return render_template('settings.html', **ctx)


@main_bp.route('/api/weights', methods=['POST'])
def api_weights():
    """更新合成权重或捕捉权重"""
    cfg = ensure_config()
    data = request.get_json(silent=True) or {}
    updated = False

    # 更新平台设置（权重/名称/颜色）
    if 'synth_weights' in data:
        for mid, w in data['synth_weights'].items():
            if mid in cfg.get('detection_methods', {}):
                cfg['detection_methods'][mid]['weight'] = float(w)
                if mid in detection_methods:
                    detection_methods[mid]['weight'] = float(w)
                updated = True
    if 'names' in data:
        for mid, name in data['names'].items():
            if mid in cfg.get('detection_methods', {}) and name.strip():
                cfg['detection_methods'][mid]['name'] = name.strip()
                if mid in detection_methods:
                    detection_methods[mid]['name'] = name.strip()
                updated = True
    if 'colors' in data:
        for mid, color in data['colors'].items():
            if mid in cfg.get('detection_methods', {}) and color.strip():
                cfg['detection_methods'][mid]['color'] = color.strip()
                if mid in detection_methods:
                    detection_methods[mid]['color'] = color.strip()
                updated = True

    # 更新捕捉权重
    if 'capture_weights' in data:
        cw = cfg.get('capture_weights', {})
        for k, v in data['capture_weights'].items():
            if k in cw:
                cw[k] = float(v)
        cfg['capture_weights'] = cw
        updated = True

    if updated:
        save_config(cfg)
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': '无有效更新'}), 400


@main_bp.route('/api/theme', methods=['GET', 'POST'])
def api_theme():
    """读取或更新主题设置"""
    cfg = ensure_config()
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        new_theme = data.get('theme', 'dark')
        if new_theme not in ('dark', 'light'):
            return jsonify({'success': False, 'error': 'theme must be dark or light'}), 400
        cfg['theme'] = new_theme
        save_config(cfg)
        return jsonify({'success': True, 'theme': new_theme})
    return jsonify({'theme': cfg.get('theme', 'dark')})
