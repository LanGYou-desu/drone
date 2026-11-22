"""
配置文件管理 — 读写 config.json

配置文件结构:
  ai:                  AI 大模型配置（支持 OpenAI 兼容接口）
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

# 项目根目录（从本文件向上 4 级：core/config/config_manager.py）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PROJECT_ROOT = _PROJECT_ROOT  # 公开别名，供其他模块使用
# 配置文件路径（绝对路径，不依赖 cwd）
CONFIG_PATH: str = os.path.join(_PROJECT_ROOT, 'config.json')
CONFIG_TEMPLATE: str = os.path.join(_PROJECT_ROOT, 'templates', 'config_template.json')

# ---- 默认配置 ----
DEFAULT_CONFIG: dict[str, Any] = {
    'ai': {
        'provider': 'siliconflow',
        'api_key': 'your-api-key-here',
        'url': 'https://api.siliconflow.cn/v1/chat/completions',
        'model': 'Qwen/Qwen2.5-7B-Instruct',
    },
    'detection_methods': {
        'visible': {
            'name': '可见光',
            'color': '#ff6b6b',
            'visible': True,
            'weight': 1.0,
            'enabled': True,
        },
        'infrared': {
            'name': '红外',
            'color': '#4ecdc4',
            'visible': True,
            'weight': 1.0,
            'enabled': True,
        },
        'radar': {
            'name': '雷达',
            'color': '#ffe66d',
            'visible': True,
            'weight': 1.0,
            'enabled': True,
        },
        'self': {
            'name': '自选',
            'color': '#FF9500',
            'visible': True,
            'weight': 1.0,
            'enabled': True,
        },
        'synthetic': {
            'name': '综合',
            'color': '#4fc3f7',
            'visible': True,
            'enabled': True,
        },
    },
    'prediction_settings': {
        # KinFlow 原生视界为 k_max × sample_dt = 1 s，之后按末速度线性延伸；
        # 默认给 10 点 × 0.2 s（2 s）既覆盖真实采样间隔，也不超出模型能力太多。
        'min_points': 3,
        'max_points': 60,
        'default_points': 10,
        'time_step': 0.2,
    },
    'theme': 'dark',
    'camera_speed': 0.12,
    'capture_weights': {
        'height': 1.0,
        'speed': 1.0,
        'acceleration': 1.0,
        'curvature': 1.0,
    },
    'drone_dynamics': {
        'g': 9.81,
        'v_h_max': 20.0,
        'v_v_up': 5.0,
        'v_v_down': 3.0,
        'max_tilt': 35.0,
        'max_alt': 120.0,
        'min_alt': 1.0,
        'thrust_max': 25.0,
        'thrust_hover': 9.81,
    },
    'training': {
        # 数据窗口
        'ctx_len': 40,
        'tgt_len': 20,
        'batch_size': 32,
        'num_workers': 4,
        # 优化（两阶段: 联合训练 + 走廊噪声微调）
        'lr_stage1': 0.001,
        'lr_stage2': 0.0001,
        'weight_decay': 0.0001,
        'epochs_stage1': 60,
        'epochs_stage2': 20,
        'warmup_epochs_s1': 5,
        'warmup_epochs_s2': 0,
        'warmup_start_factor': 0.1,
        'grad_clip': 1.0,
        'checkpoint_interval': 10,
        # 损失权重（λ_c / λ_p）
        'lambda_corridor': 0.5,
        'lambda_physics': 0.01,
        'lambda_projector': 0.1,
        'lambda_sample': 0.3,
        'projector_in_graph': True,
        'sigma_stage2': 0.1,
        # 编码器
        'd_feat': 64,
        'd_context': 128,
        'n_head': 4,
        'n_layers': 4,
        'dim_feedforward': 256,
        'dropout': 0.1,
        # 运动学走廊
        'corridor_hidden': 128,
        # 走廊流
        'd_z': 128,
        'flow_hidden': 128,
        'flow_hidden_layers': 3,
        'tau_emb_dim': 16,
        'time_emb_dim': 16,
        'k_max': 20,
        # 采样与物理边界（对齐 drone_dynamics）
        'n_euler_steps': 4,
        'sigma': 0.0,
        'v_h_max': 20.0,
        'v_up_max': 5.0,
        'v_down_max': 3.0,
        'y_min': 1.0,
        'y_max': 120.0,
        'max_tilt': 35.0,
        'g': 9.81,
        'n_project_iters': 3,
        'project': True,
        'eval_samples': 8,
        'eval_sigma': 0.1,
        'seed': 42,
        'time_jitter': 0.35,
    },
    'kinflow_model': {
        'enabled': True,
        'v_h_max': 20.0,
        'v_up_max': 5.0,
        'v_down_max': 3.0,
        'max_tilt': 35.0,
        'y_min': 1.0,
        'y_max': 120.0,
        'n_euler_steps': 4,
        'sigma': 0.0,
        'project': True,
        'device': 'cpu',
    },
    'synchronization': {
        'hz': 10.0,
        'max_offset_s': 0.6,
        'max_sample_gap_s': 0.5,
    },
    'sensor_fusion': {
        'gate_m': 3.0,
        'default_sigma': 1.5,
        'sigmas': {'visible': 1.0, 'infrared': 1.5, 'radar': 0.6, 'self': 2.0},
    },
    'reconstruction': {
        'max_gap_s': 0.8,
        'smooth_window': 5,
        'outlier_sigma': 3.5,
    },
    'assessment': {
        'zone': {'center': [8.0, 12.0, 4.0], 'radius': 6.0, 'y_min': 5.0, 'y_max': 25.0},
        'thresholds': [75.0, 50.0, 25.0],
        'weights': {'proximity': 0.30, 'closing': 0.25, 'altitude': 0.10,
                    'maneuver': 0.10, 'prediction': 0.20, 'dwell': 0.05},
        'prediction_horizon_s': 2.0,
        'prediction_points': 10,
    },
    'detection': {
        'model': 'models/yolo/yolov8n.pt',
        'confidence_threshold': 0.5,
        'nms_threshold': 0.45,
        'frame_interval': 3,
        'tracker': 'bytetrack',
        'input_width': 640,
        'input_height': 640,
        'device': 'cpu',
        'auto_save': False,
        'target_classes': [0],
        'target_class_id': 0,
    },
    'platforms': {
        'visible': {
            'focal_length_px': 0, 'baseline': 1, 'fov_horizontal': 90,
            'fov_vertical': 60, 'resolution_width': 1920, 'resolution_height': 1080,
            'pos_x': 0, 'pos_y': 0, 'pos_z': 0, 'pitch': 0, 'yaw': 0, 'roll': 0,
        },
        'infrared': {
            'focal_length_px': 0, 'baseline': 1.2, 'fov_horizontal': 60,
            'fov_vertical': 45, 'resolution_width': 640, 'resolution_height': 480,
            'pos_x': 2, 'pos_y': 1.5, 'pos_z': 0, 'pitch': 5, 'yaw': 0, 'roll': 0,
            'convergence_angle': 0, 'tilt_angle': 5,
        },
        'radar': {
            'focal_length_px': 0, 'baseline': 2, 'fov_horizontal': 120,
            'fov_vertical': 90, 'resolution_width': 1024, 'resolution_height': 768,
            'pos_x': -2, 'pos_y': 1.5, 'pos_z': 0, 'pitch': 0, 'yaw': 90, 'roll': 0,
            'convergence_angle': 0, 'tilt_angle': 0,
        },
        'self': {
            'focal_length_px': 0, 'baseline': 1, 'fov_horizontal': 90,
            'fov_vertical': 60, 'resolution_width': 1920, 'resolution_height': 1080,
            'pos_x': 0, 'pos_y': 1.5, 'pos_z': 0, 'pitch': 0, 'yaw': 0, 'roll': 0,
            'convergence_angle': 0, 'tilt_angle': 0,
        },
    },
}


def _validate_config(cfg: dict) -> dict:
    """校验并补全配置字段"""
    # 确保顶层 key 存在
    for key in DEFAULT_CONFIG:
        if key not in cfg:
            cfg[key] = DEFAULT_CONFIG[key]

    # 确保 ai 子字段
    for key in DEFAULT_CONFIG['ai']:
        if key not in cfg['ai']:
            cfg['ai'][key] = DEFAULT_CONFIG['ai'][key]

    # 确保 detection_methods 子字段
    for mid in DEFAULT_CONFIG['detection_methods']:
        if mid not in cfg['detection_methods']:
            cfg['detection_methods'][mid] = dict(DEFAULT_CONFIG['detection_methods'][mid])
        else:
            for field in ['name', 'color', 'visible', 'weight', 'enabled']:
                if field not in cfg['detection_methods'][mid]:
                    default_val = DEFAULT_CONFIG['detection_methods'][mid].get(field)
                    if default_val is not None:
                        cfg['detection_methods'][mid][field] = default_val

    # 确保 prediction_settings 子字段
    for key in DEFAULT_CONFIG['prediction_settings']:
        if key not in cfg.get('prediction_settings', {}):
            if 'prediction_settings' not in cfg:
                cfg['prediction_settings'] = {}
            cfg['prediction_settings'][key] = DEFAULT_CONFIG['prediction_settings'][key]

    # 确保 theme 和 camera_speed 字段存在
    if 'theme' not in cfg:
        cfg['theme'] = DEFAULT_CONFIG['theme']
    if 'camera_speed' not in cfg:
        cfg['camera_speed'] = DEFAULT_CONFIG['camera_speed']

    # 确保 capture_weights 存在
    if 'capture_weights' not in cfg:
        cfg['capture_weights'] = dict(DEFAULT_CONFIG['capture_weights'])
    for k in DEFAULT_CONFIG['capture_weights']:
        if k not in cfg['capture_weights']:
            cfg['capture_weights'][k] = DEFAULT_CONFIG['capture_weights'][k]

    return cfg


def ensure_config(config_path: str = CONFIG_PATH) -> dict:
    """
    确保配置文件存在，若不存在则从模板或默认配置创建
    返回经过校验的完整配置字典
    """
    if not os.path.exists(config_path):
        # 优先从模板文件复制
        if os.path.exists(CONFIG_TEMPLATE):
            import shutil
            try:
                shutil.copy(CONFIG_TEMPLATE, config_path)
                print(f'[OK] 从模板创建配置文件 {config_path}')
            except (IOError, OSError) as e:
                print(f'[WARN] 无法从模板复制 ({e})，使用默认配置')
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(DEFAULT_CONFIG, f, indent=4, ensure_ascii=False)
                return DEFAULT_CONFIG.copy()
        else:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(DEFAULT_CONFIG, f, indent=4, ensure_ascii=False)
            print(f'[OK] 已创建默认配置文件 {config_path}')
            print('  请编辑文件填入硅基流动 API Key 后重启程序。')
            return DEFAULT_CONFIG.copy()

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f'[ERR] 配置文件已损坏 ({e})，使用默认配置')
        return DEFAULT_CONFIG.copy()

    # 校验并补全
    cfg = _validate_config(cfg)
    return cfg


def save_config(cfg: dict, config_path: str = CONFIG_PATH):
    """原子写入配置到文件，避免部分写入导致配置损坏"""
    import tempfile
    dirname = os.path.dirname(config_path) or '.'
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', dir=dirname,
            delete=False, suffix='.tmp'
        ) as tmp:
            json.dump(cfg, tmp, indent=4, ensure_ascii=False)
            tmp_name = tmp.name
        os.replace(tmp_name, config_path)
    except (IOError, OSError) as e:
        print(f'[ERR] 无法保存配置到 {config_path}: {e}')
        raise RuntimeError(f'配置保存失败: {e}') from e
