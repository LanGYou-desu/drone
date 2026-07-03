"""
HTTP 视图层 — Views

轨迹识别模块的页面路由与 API 端点。

蓝图:
  pages_bp          — 页面路由 (/, /settings, /history)
  api_detection_bp  — 检测 API (start/stop/status/tracks/preview/history/config)
"""
from flask import Blueprint

pages_bp = Blueprint('recog_pages', __name__)
api_detection_bp = Blueprint('api_detection', __name__, url_prefix='/api/detection')

from trajectory_recognition.views import pages, api_detection  # noqa: E402,F401 — 注册路由
