"""
轨迹识别 — Flask 应用工厂（框架）

可独立启动，也可被根目录 main.py 统一加载。
"""
import os
import sys

from flask import Flask, jsonify

# 确保项目根目录在 sys.path 且为工作目录（支持任意 cwd 启动）
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_MODULE_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)


def create_app() -> Flask:
    """创建轨迹识别 Flask 应用"""
    app = Flask(__name__)

    # ---- 基础路由 ----
    @app.route('/')
    def index():
        return jsonify({
            'module': 'trajectory_recognition',
            'status': 'running',
            'version': '0.1.0',
            'endpoints': [
                'GET  /',
                'GET  /health',
                'GET  /api/status',
            ],
        })

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
