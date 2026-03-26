"""
配置文件管理模块
仅用于存储 API 密钥、模型参数、检测手段元信息（名称、颜色、可见性）
轨迹数据不保存在此文件中，从 data/fact/*.dat 和用户上传文件中加载
"""
import json
import os

CONFIG_PATH = 'config.json'

DEFAULT_CONFIG = {
    "siliconflow": {
        "api_key": "your-siliconflow-api-key-here",  # 请替换为真实 API Key
        "url": "https://api.siliconflow.cn/v1/chat/completions",
        "model": "Qwen/Qwen2.5-7B-Instruct"
    },
    "detection_methods": {
        "visible": {
            "name": "可见光",
            "color": "#ff6b6b",
            "visible": True
        },
        "infrared": {
            "name": "红外",
            "color": "#4ecdc4",
            "visible": True
        },
        "radar": {
            "name": "雷达",
            "color": "#ffe66d",
            "visible": True
        }
    },
    "prediction_settings": {          # 新增预测配置
        "min_points": 1,
        "max_points": 20,
        "default_points": 6
    }
}


def ensure_config():
    """
    确保配置文件存在，若不存在则创建默认配置
    返回配置字典
    """
    if not os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(DEFAULT_CONFIG, f, indent=4, ensure_ascii=False)
        print(f"已创建默认配置文件 {CONFIG_PATH}，请编辑并填入正确的硅基流动 API Key 后重启程序。")
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_config(config):
    """保存配置到文件"""
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)