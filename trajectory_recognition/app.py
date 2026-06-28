"""
轨迹识别 — Flask 应用工厂（框架）

可独立启动，也可被根目录 main.py 统一加载。
"""
import os
import sys

from flask import Flask, jsonify, render_template, send_from_directory

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_MODULE_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

# 静态资源目录：优先本模块，回退到重建模块
_OWN_STATIC = os.path.join(_MODULE_DIR, 'frontend', 'static')
_SHARED_STATIC = os.path.join(_PROJECT_ROOT, 'trajectory_reconstruction', 'frontend', 'static')


def create_app() -> Flask:
    """创建轨迹识别 Flask 应用"""
    app = Flask(
        __name__,
        template_folder=os.path.join(_MODULE_DIR, 'frontend', 'pages'),
        static_folder=None,
    )

    @app.route('/static/<path:filename>')
    def serve_static(filename):
        own = os.path.join(_OWN_STATIC, filename)
        if os.path.isfile(own):
            return send_from_directory(_OWN_STATIC, filename)
        return send_from_directory(_SHARED_STATIC, filename)

    # ---- 页面 ----
    @app.route('/')
    def index():
        return render_template('index.html')

    # ---- API ----
    @app.route('/health')
    def health():
        return jsonify({'status': 'ok'})

    @app.route('/api/status')
    def status():
        return jsonify({
            'module': 'trajectory_recognition',
            'features': 'pending',
            'models': 'pending',
            'classifier': 'pending',
        })

    return app
