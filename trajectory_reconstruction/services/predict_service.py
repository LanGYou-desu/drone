"""
预测编排服务 — 参数校验 + 调用预测算法 + 结果持久化

优先使用 Phy-ODE-Diffusion 混合模型，不可用时回退到线性外推。
"""
from trajectory_reconstruction.core.state import detection_methods
from trajectory_reconstruction.core.config.config_manager import ensure_config
from trajectory_reconstruction.core.prediction.prediction import (
    generate_prediction,
    generate_prediction_hybrid,
    is_hybrid_model_available,
)
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

    优先使用 Phy-ODE-Diffusion 混合模型（若权重文件存在），
    否则回退到线性外推。综合轨迹(synthetic)不参与预测。
    """
    data = detection_methods.get(method_id)
    if not data or not data.get('enabled', True) or len(data.get('points', [])) < 2:
        return None

    num_points, time_step = clamp_params(num_points, time_step)
    points = data['points']
    timestamps = data.get('timestamps', [])

    # 优先使用混合模型（仅对非 synthetic 平台）
    if method_id != 'synthetic' and is_hybrid_model_available():
        cfg = ensure_config()
        device = cfg.get('hybrid_model', {}).get('device', 'cpu')
        pred_points, pred_times = generate_prediction_hybrid(
            points, timestamps, num_points, time_step, device=device,
        )
    else:
        pred_points, pred_times = generate_prediction(
            points, timestamps, num_points, time_step,
        )

    return {
        'prediction': pred_points,
        'pred_times': pred_times,
    }


def predict_all(num_points: int = 6, time_step: float | None = None) -> dict:
    """
    对所有启用平台进行预测，各平台预测统一对齐到最晚结束时间
    """
    num_points, time_step = clamp_params(num_points, time_step)

    # 找到所有平台中事实数据的最后时间戳
    max_end = 0.0
    for mid, data in detection_methods.items():
        if mid == 'synthetic':
            continue
        if data.get('enabled', True) and len(data.get('points', [])) >= 2:
            ts = data.get('timestamps', [])
            if ts and ts[-1] > max_end:
                max_end = ts[-1]

    # 统一预测时间数组（锚点 + N 个预测点，等间隔）
    pred_times_unified = [round(max_end + i * time_step, 6) for i in range(num_points + 1)]

    results = {}
    for mid, data in detection_methods.items():
        if mid == 'synthetic':
            continue
        if not data.get('enabled', True):
            continue
        pts = data.get('points', [])
        if len(pts) < 2:
            continue
        result = predict_single(mid, num_points, time_step)
        if result and result['prediction']:
            # 覆盖预测时间为统一时间数组，所有平台预测段完全对齐
            result['pred_times'] = pred_times_unified
            save_predict_data(mid, result['prediction'], pred_times_unified)
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

    # 用最长平台的预测长度统一所有轨迹
    max_pred_len = 0
    ref_times = None
    for mid, result in results.items():
        pred = result.get('prediction', [])
        if len(pred) > max_pred_len:
            max_pred_len = len(pred)
            ref_times = result.get('pred_times', [])[:max_pred_len]

    if max_pred_len < 2 or not ref_times:
        return [], []

    sorted_ts = ref_times

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
