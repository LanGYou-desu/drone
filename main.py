"""
鹰眼长空 — 无人机智能监测系统 主入口
启动 Flask 后端 + Windows 原生窗口（pywebview）
"""
import os
import sys
import threading
import time

from flask import Flask

from modules.config.config_manager import ensure_config
from modules.services.state import init_from_config
from modules.services.data_service import initialize_data
from modules.routes import register_blueprints, register_error_handlers


def create_app() -> Flask:
    """Flask 应用工厂"""
    app = Flask(__name__)

    # 确保配置和数据目录
    ensure_config()
    os.makedirs('data/fact', exist_ok=True)
    os.makedirs('data/predict', exist_ok=True)
    os.makedirs('data/backup', exist_ok=True)

    # 初始化共享状态
    init_from_config()
    initialize_data()

    # 注册蓝图和错误处理
    register_blueprints(app)
    register_error_handlers(app)

    return app


def run_flask(app: Flask, host: str = '127.0.0.1', port: int = 5000):
    """后台线程中运行 Flask"""
    app.run(host=host, port=port, debug=False, use_reloader=False)


if __name__ == '__main__':
    # 创建应用
    flask_app = create_app()

    # 检查是否需要无窗口模式（命令行参数 --no-window）
    if '--no-window' in sys.argv:
        print(f'🚀 服务启动: http://127.0.0.1:5000')
        run_flask(flask_app)
    else:
        import webview

        # 后台启动 Flask
        flask_thread = threading.Thread(
            target=run_flask, args=(flask_app,), daemon=True
        )
        flask_thread.start()
        time.sleep(2)

        # 创建桌面窗口
        webview.create_window(
            title='鹰眼长空 — 低空无人机智能监测系统',
            url='http://127.0.0.1:5000',
            width=1280,
            height=720,
            resizable=True,
            fullscreen=False,
        )
        webview.start()
