"""
共享运行时状态管理

维护检测手段数据（轨迹、元信息）的全局内存缓存。
路由蓝图通过此模块共享状态，避免循环导入。
"""
import threading
import time
from trajectory_reconstruction.core.config.config_manager import ensure_config

# 启动版本号，用于前端缓存刷新
APP_VERSION = str(int(time.time()))

_config = ensure_config()

# { methodId: { name, color, visible, points, timestamps } }
detection_methods: dict[str, dict] = {}
_state_lock = threading.RLock()

# AI 大模型配置（兼容旧 siliconflow 字段）
_ai_cfg = _config.get('ai', _config.get('siliconflow', {}))
AI_API_KEY: str = _ai_cfg.get('api_key', '')
AI_URL: str = _ai_cfg.get('url', '')
AI_MODEL: str = _ai_cfg.get('model', '')


def get_methods_snapshot() -> dict:
    """返回检测方法的线程安全浅拷贝快照"""
    with _state_lock:
        return {mid: dict(data) for mid, data in detection_methods.items()}


def update_method(method_id: str, **kwargs):
    """线程安全地更新单个检测方法的属性"""
    with _state_lock:
        if method_id in detection_methods:
            detection_methods[method_id].update(kwargs)


METHOD_ORDER = ['visible', 'infrared', 'radar', 'self', 'synthetic']

def init_from_config():
    """从配置文件初始化 detection_methods 元信息（固定顺序）"""
    cfg = ensure_config()
    detection_methods.clear()
    for mid in METHOD_ORDER:
        if mid in cfg['detection_methods']:
            data = cfg['detection_methods'][mid]
            detection_methods[mid] = _make_method(data)
    # 配置中有但不在固定列表中的
    for mid, data in cfg['detection_methods'].items():
        if mid not in detection_methods:
            detection_methods[mid] = _make_method(data)


def _make_method(data: dict) -> dict:
    result = {
        'name': data.get('name', ''),
        'color': data.get('color', '#999999'),
        'visible': data.get('visible', True),
        'enabled': data.get('enabled', True),
        'points': [],
        'timestamps': [],
    }
    if 'weight' in data:
        result['weight'] = data['weight']
    return result
