"""
轨迹识别 — Flask 应用工厂

基于 YOLO 模型的无人机视频检测系统。
可独立启动（端口 5001），也可被根目录 main.py 统一加载。
"""
import os
import sys

from flask import Flask, jsonify, send_from_directory

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
    shared_templates = os.path.join(_PROJECT_ROOT, 'templates', 'frontend', 'shared')
    module_pages = os.path.join(_MODULE_DIR, 'frontend', 'pages')
    shared_static = os.path.join(_PROJECT_ROOT, 'templates', 'frontend', 'static')
    app = Flask(
        __name__,
        template_folder=module_pages,
        static_folder=None,
    )
    app.jinja_loader.searchpath.insert(0, shared_templates)

    # ── 静态文件 ──────────────────────────────────

    @app.route('/static/<path:filename>')
    def serve_static(filename):
        own = os.path.join(_OWN_STATIC, filename)
        if os.path.isfile(own):
            return send_from_directory(_OWN_STATIC, filename)
        return send_from_directory(shared_static, filename)

    @app.after_request
    def _no_cache(response):
        """禁止浏览器缓存"""
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response

    # ── 启动时清理临时上传 ──────────────────────

    import glob
    uploads = os.path.join(_PROJECT_ROOT, 'data', 'uploads')
    if os.path.isdir(uploads):
        for f in glob.glob(os.path.join(uploads, 'temp_*')):
            try:
                os.remove(f)
            except Exception:
                pass

    # ── 注册蓝图 ──────────────────────────────────

    from trajectory_recognition.views import pages_bp, api_detection_bp
    app.register_blueprint(pages_bp)
    app.register_blueprint(api_detection_bp)

    # ── 健康检查 ──────────────────────────────────

    @app.route('/health')
    def health():
        return jsonify({'status': 'ok'})

    @app.route('/api/status')
    def status():
        return jsonify({
            'module': 'trajectory_recognition',
            'features': 'stub',
            'models': 'stub',
            'classifier': 'stub',
            'detection': 'stub',
        })

    return app
