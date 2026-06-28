"""
共享运行时状态管理

维护检测手段数据（轨迹、元信息）的全局内存缓存。
路由蓝图通过此模块共享状态，避免循环导入。
"""
from trajectory_reconstruction.core.config.config_manager import ensure_config

_config = ensure_config()

# { methodId: { name, color, visible, points, timestamps } }
detection_methods: dict[str, dict] = {}

# API 配置（硅基流动）
SF_API_KEY: str = _config['siliconflow']['api_key']
SF_URL: str = _config['siliconflow']['url']
SF_MODEL: str = _config['siliconflow']['model']


def init_from_config():
    """从配置文件初始化 detection_methods 元信息"""
    cfg = ensure_config()
    detection_methods.clear()
    for mid, data in cfg['detection_methods'].items():
        detection_methods[mid] = {
            'name': data.get('name', ''),
            'color': data.get('color', '#999999'),
            'visible': data.get('visible', True),
            'points': [],
            'timestamps': [],
        }
