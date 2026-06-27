"""
预测编排服务 — 参数校验 + 调用预测算法 + 结果持久化
"""
from modules.state import detection_methods
from modules.config.config_manager import ensure_config
from modules.predict.prediction import generate_prediction
from modules.data.data_loader import save_predict_data


def get_predict_config() -> dict:
    """获取预测配置（含默认值）"""
    cfg = ensure_config()
    return cfg.get('prediction_settings', {
        'min_points': 1,
        'max_points': 20,
        'default_points': 6,
        'time_step': 0.5,
    })


def clamp_params(num_points: int, time_step: float | None) -> tuple[int, float]:
    """
    参数约束与默认值填充
    返回 (valid_num_points, valid_time_step)
    """
    settings = get_predict_config()
    min_pts = settings.get('min_points', 1)
    max_pts = settings.get('max_points', 20)
    default_step = settings.get('time_step', 0.5)

    num_points = max(min_pts, min(num_points, max_pts))

    if time_step is None:
        time_step = default_step
    else:
        time_step = float(time_step)
        if time_step <= 0:
            time_step = default_step

    return num_points, time_step


def predict_single(method_id: str, num_points: int = 6,
                   time_step: float | None = None) -> dict | None:
    """
    对指定平台进行预测，返回 { prediction: [[x,y,z],...], pred_times: [t,...] }
    若条件不满足返回 None
    """
    data = detection_methods.get(method_id)
    if not data or not data.get('visible') or len(data.get('points', [])) < 2:
        return None

    num_points, time_step = clamp_params(num_points, time_step)
    points = data['points']
    timestamps = data.get('timestamps', [])
    pred_points, pred_times = generate_prediction(points, timestamps, num_points, time_step)

    if pred_points:
        save_predict_data(method_id, pred_points, pred_times)

    return {
        'prediction': pred_points,
        'pred_times': pred_times,
    }


def predict_all(num_points: int = 6, time_step: float | None = None) -> dict:
    """
    对所有可见平台进行预测
    返回 { methodId: { prediction, pred_times } }
    """
    results = {}
    for mid, data in detection_methods.items():
        if data.get('visible') and len(data.get('points', [])) >= 2:
            result = predict_single(mid, num_points, time_step)
            if result and result['prediction']:
                results[mid] = result
    return results
