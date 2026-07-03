"""
轨迹重建与分析 — Flask 应用工厂

可独立启动，也可被根目录 main.py 统一加载。
"""
import os
import sys

from flask import Flask, send_from_directory

# 确保项目根目录在 sys.path 且为工作目录（支持任意 cwd 启动）
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_MODULE_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

# 静态资源（优先共享，回退模块）
_SHARED_STATIC = os.path.join(_PROJECT_ROOT, 'templates', 'frontend', 'static')
_OWN_STATIC = os.path.join(_MODULE_DIR, 'frontend', 'static')


def create_app() -> Flask:
    """创建轨迹重建与分析 Flask 应用"""
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # 避免 OpenCV+PyTorch OpenMP 冲突
    os.chdir(_PROJECT_ROOT)  # 确保工作目录正确

    shared_templates = os.path.join(_PROJECT_ROOT, 'templates', 'frontend', 'shared')
    module_pages = os.path.join(_MODULE_DIR, 'frontend', 'pages')
    app = Flask(
        __name__,
        template_folder=module_pages,
        static_folder=None,
    )
    app.jinja_loader.searchpath.insert(0, shared_templates)

    @app.route('/static/<path:filename>')
    def serve_static(filename):
        shared = os.path.join(_SHARED_STATIC, filename)
        if os.path.isfile(shared):
            return send_from_directory(_SHARED_STATIC, filename)
        return send_from_directory(_OWN_STATIC, filename)

    @app.after_request
    def _no_cache(response):
        """禁止浏览器缓存，确保每次刷新拿到最新文件"""
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response

    @app.route('/favicon.ico')
    def favicon():
        from flask import Response
        svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16">'
               '<polygon points="8,1 14,4 14,10 8,13 2,10 2,4" fill="none" stroke="#58a6ff" stroke-width="1.2"/>'
               '<line x1="8" y1="4" x2="8" y2="10" stroke="#58a6ff" stroke-width="1.2"/>'
               '<line x1="5" y1="5.5" x2="11" y2="8.5" stroke="#58a6ff" stroke-width="1.2"/>'
               '<line x1="11" y1="5.5" x2="5" y2="8.5" stroke="#58a6ff" stroke-width="1.2"/>'
               '</svg>')
        return Response(svg, mimetype='image/svg+xml')

    # 延迟导入，避免循环依赖
    from trajectory_reconstruction.core.config.config_manager import ensure_config
    from trajectory_reconstruction.core.state import init_from_config
    from trajectory_reconstruction.services.data_service import initialize_data
    from trajectory_reconstruction.views import register_blueprints, register_error_handlers

    # 确保配置和数据目录
    ensure_config()
    data_root = os.path.join(_PROJECT_ROOT, 'data')
    os.makedirs(os.path.join(data_root, 'fact'), exist_ok=True)
    os.makedirs(os.path.join(data_root, 'predict'), exist_ok=True)
    os.makedirs(os.path.join(data_root, 'backup'), exist_ok=True)

    # 清理旧版备份文件
    from trajectory_reconstruction.services.backup_service import migrate_legacy
    migrate_legacy()

    # 初始化共享状态
    init_from_config()
    initialize_data()

    # 注册蓝图和错误处理
    register_blueprints(app)
    register_error_handlers(app)

    return app
