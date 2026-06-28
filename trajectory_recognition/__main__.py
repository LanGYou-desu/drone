"""
轨迹识别 — 独立启动入口

用法:
  python -m trajectory_recognition              # 桌面窗口模式
  python -m trajectory_recognition --headless   # 纯 HTTP 模式
"""
import sys
import threading
import time

from trajectory_recognition.app import create_app


def run_flask(app, host='127.0.0.1', port=5001):
    app.run(host=host, port=port, debug=False, use_reloader=False)


if __name__ == '__main__':
    app = create_app()

    if '--headless' in sys.argv:
        print(f'[recog] http://127.0.0.1:5001')
        run_flask(app)
    else:
        import webview

        flask_thread = threading.Thread(
            target=run_flask, args=(app,), daemon=True
        )
        flask_thread.start()
        time.sleep(2)

        webview.create_window(
            title='鹰眼长空 — 轨迹识别',
            url='http://127.0.0.1:5001',
            width=1280, height=800,
            resizable=True, fullscreen=False,
        )
        webview.start()
