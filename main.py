"""
无人机轨迹预测系统 - 主入口
启动 Flask 后端线程，并创建 Windows 原生窗口
"""
import os
import threading
import time
import webview
from flask import Flask
from modules.config.config_manager import ensure_config
from modules.routes.main import main_bp
from modules.routes.analysis import analysis_bp

app = Flask(__name__)

# 确保配置文件存在（仅用于 API 密钥等配置）
config = ensure_config()

# 注册蓝图
app.register_blueprint(main_bp)
app.register_blueprint(analysis_bp, url_prefix='/analysis')


def run_flask():
    """在后台线程中运行 Flask 开发服务器"""
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)


if __name__ == '__main__':
    # 确保数据目录存在
    os.makedirs('data/fact', exist_ok=True)
    os.makedirs('data/predict', exist_ok=True)

    # 启动 Flask 线程
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    # 等待 Flask 完全启动
    time.sleep(2)

    # 创建原生窗口
    webview.create_window(
        title='无人机轨迹预测系统',
        url='http://127.0.0.1:5000',
        width=1280,
        height=720,
        resizable=True,
        fullscreen=False
    )
    webview.start(gui='edgechromium')  # 使用 Edge Chromium 后端，避免 pythonnet 问题