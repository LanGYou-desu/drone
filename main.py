from flask import Flask
from config_manager import ensure_config
from routes.main import main_bp
from routes.analysis import analysis_bp
import threading
import webview

app = Flask(__name__)

# 确保配置文件存在
config = ensure_config()

# 注册蓝图
app.register_blueprint(main_bp)
app.register_blueprint(analysis_bp, url_prefix='/analysis')

def run_flask():
    """在后台线程中运行 Flask 应用"""
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)

if __name__ == '__main__':
    # 启动 Flask 后台线程
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    # 等待 Flask 启动（2秒足够）
    import time
    time.sleep(2)

    # 创建原生窗口
    webview.create_window(
        title='无人机轨迹预测系统',
        url='http://127.0.0.1:5000',
        width=1280,
        height=720,
        resizable=True,
        fullscreen=False,
        confirm_close=True  # 关闭时确认
    )
    webview.start()