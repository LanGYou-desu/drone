"""
页面路由 — Page Routes

轨迹识别模块的前端页面渲染。
"""

from flask import render_template

from trajectory_recognition.views import pages_bp


@pages_bp.route('/')
def index():
    """检测主页面 — 视频上传、实时检测预览、结果展示"""
    return render_template('index.html')


@pages_bp.route('/settings')
def settings():
    """检测设置页面 — 模型选择、参数配置"""
    return render_template('settings.html')


@pages_bp.route('/history')
def history():
    """检测历史页面 — 查看历史检测记录"""
    return render_template('history.html')
