"""
鹰眼长空 — 无人机智能监测系统 统一入口

用法:
  python main.py               # 默认同时启动两个模块
  python main.py recon         # 仅轨迹重建与分析
  python main.py recog         # 仅轨迹识别
  python main.py --headless    # 纯 HTTP 模式

也可独立启动各模块:
  python -m trajectory_reconstruction --headless   # → :5000
  python -m trajectory_recognition --headless       # → :5001
  python -m trajectory_recognition                  # 桌面窗口
"""
import os
import subprocess
import sys
import threading
import time


def _cleanup_ports():
    """启动前终止占用目标端口的残留进程，确保每次启动都是最新版本"""
    for port in (5000, 5001):
        try:
            if sys.platform == 'win32':
                result = subprocess.run(
                    ['netstat', '-ano'], capture_output=True, text=True
                )
                for line in result.stdout.splitlines():
                    if f':{port}' in line and 'LISTENING' in line:
                        parts = line.strip().split()
                        pid = parts[-1]
                        subprocess.run(
                            ['taskkill', '/F', '/PID', pid],
                            capture_output=True,
                        )
                        print(f'[cleanup] 终止残留进程 PID={pid} (端口 {port})')
            else:
                result = subprocess.run(
                    ['lsof', '-ti', f':{port}'], capture_output=True, text=True
                )
                for pid in result.stdout.strip().splitlines():
                    if pid:
                        os.kill(int(pid), 9)
                        print(f'[cleanup] 终止残留进程 PID={pid} (端口 {port})')
        except Exception:
            pass


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

    _cleanup_ports()

    if 'recog' in args and 'recon' not in args:
        start_recog()
    elif 'recon' in args:
        start_recon(headless=headless)
    else:
        # 默认同时启动两个模块
        threading.Thread(target=start_recog, daemon=True).start()
        time.sleep(1)
        start_recon(headless=headless)
