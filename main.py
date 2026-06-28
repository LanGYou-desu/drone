"""
鹰眼长空 — 无人机智能监测系统 统一入口

用法:
  python main.py               # 轨迹重建与分析（桌面窗口）
  python main.py --headless    # 轨迹重建与分析（纯 HTTP）
  python main.py recon         # 轨迹重建与分析
  python main.py recog         # 轨迹识别
  python main.py all           # 同时启动两个模块

也可独立启动各模块:
  python -m trajectory_reconstruction --headless   # → :5000
  python -m trajectory_recognition --headless       # → :5001
  python -m trajectory_recognition                  # 桌面窗口
"""
import sys
import threading
import time


def start_recon(headless=False):
    """启动轨迹重建与分析模块"""
    from trajectory_reconstruction.app import create_app
    app = create_app()

    if headless:
        print('[recon] http://127.0.0.1:5000')
        app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)
    else:
        import webview

        def _run():
            app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)

        flask_thread = threading.Thread(target=_run, daemon=True)
        flask_thread.start()
        time.sleep(2)

        webview.create_window(
            title='鹰眼长空 — 轨迹重建与分析',
            url='http://127.0.0.1:5000',
            width=1280, height=800,
            resizable=True, fullscreen=False,
        )
        webview.start()


def start_recog():
    """启动轨迹识别模块"""
    from trajectory_recognition.app import create_app
    app = create_app()
    print('[recog] http://127.0.0.1:5001')
    app.run(host='127.0.0.1', port=5001, debug=False, use_reloader=False)


if __name__ == '__main__':
    args = set(sys.argv[1:])
    headless = '--headless' in args

    if 'all' in args:
        threading.Thread(target=start_recog, daemon=True).start()
        time.sleep(1)
        start_recon(headless=headless)
    elif 'recog' in args:
        start_recog()
    else:
        # 默认或无参数 / recon → 轨迹重建与分析
        start_recon(headless=headless)
