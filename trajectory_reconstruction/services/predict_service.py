"""
预测编排服务 — 参数校验 + 调用预测算法 + 结果持久化
"""
from trajectory_reconstruction.core.state import detection_methods
from trajectory_reconstruction.core.config.config_manager import ensure_config
from trajectory_reconstruction.core.prediction.prediction import generate_prediction
from trajectory_reconstruction.core.io.data_loader import save_predict_data


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
    对所有可见平台进行预测，综合轨迹的预测由其他平台加权合成
    """
    results = {}
    # 先预测各独立平台（跳过综合）
    for mid, data in detection_methods.items():
        if mid == 'synthetic':
            continue
        if data.get('visible') and len(data.get('points', [])) >= 2:
            result = predict_single(mid, num_points, time_step)
            if result and result['prediction']:
                results[mid] = result
    # 合成综合预测：加权平均各平台的预测
    if 'synthetic' in detection_methods and len(results) >= 2:
        syn_pred, syn_times = _synthesize_predictions(results, num_points, time_step)
        if syn_pred:
            save_predict_data('synthetic', syn_pred, syn_times)
            results['synthetic'] = {'prediction': syn_pred, 'pred_times': syn_times}
    return results


def _synthesize_predictions(results, num_points, time_step):
    """加权合成综合预测"""
    active = {}
    for mid, result in results.items():
        m = detection_methods.get(mid)
        if m:
            w = m.get('weight', 1.0)
            if w > 0:
                active[mid] = (w, result['prediction'], result['pred_times'])

    if len(active) < 2:
        return [], []

    # 收集所有预测时间点
    all_ts = set()
    for _, _, ts in active.values():
        for t in ts:
            all_ts.add(round(t, 6))
    sorted_ts = sorted(all_ts)

    syn_points, syn_times = [], []
    for t in sorted_ts:
        wx, wy, wz, wsum = 0.0, 0.0, 0.0, 0.0
        for mid, (w, pts, ts_arr) in active.items():
            p = _lerp_pts(pts, ts_arr, t)
            if p is None:
                p = _nearest_avg_pts(pts, ts_arr, t)
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

    # 平滑
    for _ in range(2):
        s = []
        for i in range(len(syn_points)):
            if i == 0 or i == len(syn_points)-1:
                s.append(syn_points[i][:])
            else:
                s.append([(syn_points[i-1][j]+syn_points[i][j]+syn_points[i+1][j])/3 for j in range(3)])
        syn_points = s

    return syn_points, syn_times


def _lerp_pts(pts, ts_arr, t):
    """在时间点t插值"""
    if not pts or not ts_arr or len(pts) < 2:
        return None
    import bisect
    if t <= ts_arr[0]:
        return pts[0]
    if t >= ts_arr[-1]:
        return pts[-1]
    i = bisect.bisect_left(ts_arr, t)
    if i <= 0 or i >= len(ts_arr):
        return pts[i] if i < len(pts) else pts[-1]
    t0, t1 = ts_arr[i-1], ts_arr[i]
    if t1 == t0:
        return pts[i]
    r = (t - t0) / (t1 - t0)
    p0, p1 = pts[i-1], pts[i]
    return [p0[j] + (p1[j]-p0[j])*r for j in range(3)]


def _nearest_avg_pts(pts, ts_arr, t):
    """前后最近两点的平均值（_lerp_pts 的兜底）"""
    if not pts or not ts_arr or len(pts) < 2:
        return None
    prev_pt, next_pt = None, None
    for i, ti in enumerate(ts_arr):
        if ti <= t:
            prev_pt = pts[i]
        if ti >= t and next_pt is None:
            next_pt = pts[i]
    if prev_pt and next_pt:
        return [(prev_pt[j] + next_pt[j])/2 for j in range(3)]
    return prev_pt or next_pt
