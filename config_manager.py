import json
import os

CONFIG_PATH = 'config.json'

DEFAULT_CONFIG = {
    "siliconflow": {
        "api_key": "your-siliconflow-api-key-here",
        "url": "https://api.siliconflow.cn/v1/chat/completions",
        "model": "Qwen/Qwen2.5-7B-Instruct"
    },
    "detection_methods": {
        "visible": {
            "name": "可见光",
            "color": "#ff6b6b",
            "visible": True,
            "points": [
                [0, 2, 0], [2, 2.5, 1], [4, 3, 2], [6, 2.8, 3], [8, 3.2, 4], [10, 3, 5], [12, 2.5, 6]
            ]
        },
        "infrared": {
            "name": "红外",
            "color": "#4ecdc4",
            "visible": True,
            "points": [
                [0, 1.5, 0], [1.5, 2, 1.5], [3, 2.2, 3], [4.5, 2.8, 4.5], [6, 3, 6], [7.5, 2.6, 7.5]
            ]
        },
        "radar": {
            "name": "雷达",
            "color": "#ffe66d",
            "visible": True,
            "points": [
                [0, 1, 0], [2, 1.5, 1.5], [4, 2, 3], [6, 2.2, 4.5], [8, 2.5, 6], [10, 2.8, 7.5]
            ]
        }
    }
}

def ensure_config():
    """确保配置文件存在，若不存在则创建默认配置"""
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