"""
配置文件管理 — 读写 config.json

配置文件结构:
  siliconflow:         硅基流动 API 配置
    api_key            API 密钥（需用户填写）
    url                接口端点
    model              模型名称
  detection_methods:   检测手段元信息（名称/颜色/可见性）
    注：轨迹数据不存于此，从 data/fact/*.dat 加载
  prediction_settings: 预测参数约束
    min_points         最小预测点数
    max_points         最大预测点数
    default_points     默认预测点数
    time_step          预测时间步长
"""
import json
import os
from typing import Any

# 配置文件路径（相对于项目根目录）
CONFIG_PATH: str = 'config.json'

# ---- 默认配置 ----
DEFAULT_CONFIG: dict[str, Any] = {
    'siliconflow': {
        'api_key': 'your-siliconflow-api-key-here',
        'url': 'https://api.siliconflow.cn/v1/chat/completions',
        'model': 'Qwen/Qwen2.5-7B-Instruct',
    },
    'detection_methods': {
        'visible': {
            'name': '可见光',
            'color': '#FF3B30',  # Apple 红色
            'visible': True,
        },
        'infrared': {
            'name': '红外',
            'color': '#34C759',  # Apple 绿色
            'visible': True,
        },
        'radar': {
            'name': '雷达',
            'color': '#FFCC00',  # Apple 黄色
            'visible': True,
        },
    },
    'prediction_settings': {
        'min_points': 1,
        'max_points': 20,
        'default_points': 6,
        'time_step': 0.5,
    },
}


def _validate_config(cfg: dict) -> dict:
    """校验并补全配置字段"""
    # 确保顶层 key 存在
    for key in DEFAULT_CONFIG:
        if key not in cfg:
            cfg[key] = DEFAULT_CONFIG[key]

    # 确保 siliconflow 子字段
    for key in DEFAULT_CONFIG['siliconflow']:
        if key not in cfg['siliconflow']:
            cfg['siliconflow'][key] = DEFAULT_CONFIG['siliconflow'][key]

    # 确保 detection_methods 子字段
    for mid in DEFAULT_CONFIG['detection_methods']:
        if mid not in cfg['detection_methods']:
            cfg['detection_methods'][mid] = DEFAULT_CONFIG['detection_methods'][mid]
        else:
            for field in ['name', 'color', 'visible']:
                if field not in cfg['detection_methods'][mid]:
                    cfg['detection_methods'][mid][field] = DEFAULT_CONFIG['detection_methods'][mid][field]

    # 确保 prediction_settings 子字段
    for key in DEFAULT_CONFIG['prediction_settings']:
        if key not in cfg.get('prediction_settings', {}):
            if 'prediction_settings' not in cfg:
                cfg['prediction_settings'] = {}
            cfg['prediction_settings'][key] = DEFAULT_CONFIG['prediction_settings'][key]

    return cfg


def ensure_config(config_path: str = CONFIG_PATH) -> dict:
    """
    确保配置文件存在，若不存在则创建默认配置
    返回经过校验的完整配置字典
    """
    if not os.path.exists(config_path):
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(DEFAULT_CONFIG, f, indent=4, ensure_ascii=False)
        print(f'[OK] 已创建默认配置文件 {config_path}')
        print('  请编辑文件填入硅基流动 API Key 后重启程序。')
        return DEFAULT_CONFIG.copy()

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    # 校验并补全
    cfg = _validate_config(cfg)
    return cfg


def save_config(cfg: dict, config_path: str = CONFIG_PATH):
    """保存配置到文件（仅写回 JSON）"""
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=4, ensure_ascii=False)
