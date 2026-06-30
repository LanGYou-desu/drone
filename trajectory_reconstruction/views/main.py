"""
页面路由 — 多页面渲染入口
每个页面独立渲染，通过模板注入后端数据
"""
from flask import Blueprint, render_template, request, jsonify

from trajectory_reconstruction.core.state import detection_methods
from trajectory_reconstruction.core.config.config_manager import ensure_config, save_config

main_bp = Blueprint('main', __name__)


def _page_context(active: str) -> dict:
    """构建页面公共上下文"""
    cfg = ensure_config()
    return {
        'methods_data': detection_methods,
        'pred_settings': cfg.get('prediction_settings', {}),
        'active_page': active,
        'theme': cfg.get('theme', 'dark'),
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
