"""
预测编排服务 — 参数校验 + 调用预测算法 + 结果持久化
"""
from trajectory_reconstruction.core.state import detection_methods
from trajectory_reconstruction.core.config.config_manager import ensure_config
from trajectory_reconstruction.core.prediction.prediction import generate_prediction
from trajectory_reconstruction.core.io.data_loader import save_predict_data
from trajectory_reconstruction.core.math_utils import lerp_3d_extrapolate, smooth_points_3d


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
    if not data or not data.get('enabled', True) or len(data.get('points', [])) < 2:
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
    对所有启用平台进行预测，综合轨迹的预测由其他平台加权合成
    """
    results = {}
    # 先预测各独立平台（跳过综合和未启用的平台）
    for mid, data in detection_methods.items():
        if mid == 'synthetic':
            continue
        if not data.get('enabled', True):
            continue
        if len(data.get('points', [])) >= 2:
            result = predict_single(mid, num_points, time_step)
            if result and result['prediction']:
                results[mid] = result
    # 合成综合预测
    if 'synthetic' in detection_methods:
        if len(results) >= 2:
            syn_pred, syn_times = _synthesize_predictions(results, num_points, time_step)
        elif len(results) == 1:
            mid = list(results.keys())[0]
            syn_pred, syn_times = list(results[mid]['prediction']), list(results[mid]['pred_times'])
        else:
            syn_pred, syn_times = [], []
        if syn_pred:
            save_predict_data('synthetic', syn_pred, syn_times)
            results['synthetic'] = {'prediction': syn_pred, 'pred_times': syn_times}
    return results


def _synthesize_predictions(results, num_points, time_step):
    """加权合成综合预测 — 结合原始数据与预测，按时间步逐步加权"""
    active = {}
    for mid, result in results.items():
        m = detection_methods.get(mid)
        if not m or not m.get('enabled', True):
            continue
        w = m.get('weight', 1.0)
        if w <= 0:
            continue

        # 合并原始数据与预测数据（去重首点）
        fact_pts = m.get('points', [])
        fact_ts = m.get('timestamps', [])
        pred_pts = result.get('prediction', [])
        pred_ts = result.get('pred_times', [])

        combined_pts = list(fact_pts)
        combined_ts = list(fact_ts)
        if pred_pts and combined_pts:
            if abs(pred_ts[0] - combined_ts[-1]) < 1e-6:
                combined_pts.extend(pred_pts[1:])
                combined_ts.extend(pred_ts[1:])
            else:
                combined_pts.extend(pred_pts)
                combined_ts.extend(pred_ts)
        else:
            combined_pts.extend(pred_pts)
            combined_ts.extend(pred_ts)

        if len(combined_pts) >= 2:
            active[mid] = (w, combined_pts, combined_ts)

    if len(active) < 2:
        return [], []

    # 构建统一时间网格（覆盖所有平台的原始+预测数据范围）
    t_start = min(ts[0] for _, _, ts in active.values() if ts)
    t_end = max(ts[-1] for _, _, ts in active.values() if ts)
    if t_start >= t_end:
        return [], []

    # 取最细时间步作为网格分辨率
    min_dt = min(
        min((ts[i + 1] - ts[i]) for i in range(len(ts) - 1) if ts[i + 1] > ts[i])
        for _, _, ts in active.values() if len(ts) >= 2
    )
    if min_dt <= 0:
        min_dt = 0.1
    n_steps = max(2, int((t_end - t_start) / min_dt) + 1)
    sorted_ts = [round(t_start + i * (t_end - t_start) / (n_steps - 1), 6)
                 for i in range(n_steps)]

    # 按时间步加权合成
    syn_points, syn_times = [], []
    for t in sorted_ts:
        wx, wy, wz, wsum = 0.0, 0.0, 0.0, 0.0
        for mid, (w, pts, ts_arr) in active.items():
            p = lerp_3d_extrapolate(pts, ts_arr, t)
            if p is not None:
                wx += p[0] * w
                wy += p[1] * w
                wz += p[2] * w
                wsum += w
        if wsum > 0:
            syn_points.append([wx/wsum, wy/wsum, wz/wsum])
            syn_times.append(t)

    if len(syn_points) < 2:
        return [], []

    # 平滑（多平台融合才需要平滑）
    if len(active) > 1:
        syn_points = smooth_points_3d(syn_points, passes=2)

    return syn_points, syn_times
