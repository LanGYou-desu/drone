import json
from config_manager import CONFIG_PATH, save_config

def load_methods():
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config['detection_methods']

def save_methods(methods):
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)
    config['detection_methods'] = methods
    save_config(config)

def update_point(method_id, new_points):
    methods = load_methods()
    if method_id in methods:
        methods[method_id]['points'] = new_points
        save_methods(methods)
        return True
    return False