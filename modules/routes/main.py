"""
页面路由 — 多页面渲染入口
每个页面独立渲染，通过模板注入后端数据
"""
from flask import Blueprint, render_template

from modules.state import detection_methods
from modules.config.config_manager import ensure_config

main_bp = Blueprint('main', __name__)


def _page_context(active: str) -> dict:
    """构建页面公共上下文"""
    cfg = ensure_config()
    return {
        'methods_data': detection_methods,
        'pred_settings': cfg.get('prediction_settings', {}),
        'active_page': active,
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
