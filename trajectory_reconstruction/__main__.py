"""
轨迹重建与分析 — 独立启动入口

用法:
  python -m trajectory_reconstruction              # 桌面窗口模式
  python -m trajectory_reconstruction --headless  # 纯 HTTP 模式
"""
import sys
import threading
import time

from trajectory_reconstruction.app import create_app


def run_flask(app, host='127.0.0.1', port=5000):
    app.run(host=host, port=port, debug=False, use_reloader=False)


if __name__ == '__main__':
    flask_app = create_app()

    if '--headless' in sys.argv:
        print(f'[OK] 轨迹重建与分析服务: http://127.0.0.1:5000')
        run_flask(flask_app)
    else:
        import webview

        flask_thread = threading.Thread(
            target=run_flask, args=(flask_app,), daemon=True
        )
        flask_thread.start()
        time.sleep(2)

        webview.create_window(
            title='鹰眼长空 — 轨迹重建与分析',
            url='http://127.0.0.1:5000',
            width=1280, height=800,
            resizable=True, fullscreen=False,
        )
        webview.start()
