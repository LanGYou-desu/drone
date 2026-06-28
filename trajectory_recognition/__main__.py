"""
轨迹识别 — 独立启动入口

用法:
  python -m trajectory_recognition
"""
from trajectory_recognition.app import create_app

if __name__ == '__main__':
    app = create_app()
    print('[OK] 轨迹识别服务: http://127.0.0.1:5001')
    app.run(host='127.0.0.1', port=5001, debug=False, use_reloader=False)
