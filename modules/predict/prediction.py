"""
轨迹预测模块：基于历史点线性外推，生成预测轨迹点及对应时间戳
"""
def generate_prediction(points, timestamps, num_points=5, time_step=0.5):
    """
    基于最后两点线性外推预测未来轨迹，并生成对应时间戳

    参数:
        points: list of [x,y,z]，历史点坐标
        timestamps: list of float，历史点对应的时间
        num_points: int，预测点数
        time_step: float，预测点之间的时间间隔

    返回:
        pred_points: list of [x,y,z]，预测点坐标
        pred_times: list of float，预测点时间
    """
    if len(points) < 2:
        return [], []

    last = points[-1]
    second_last = points[-2]
    last_t = timestamps[-1]
    second_last_t = timestamps[-2]

    dt = last_t - second_last_t
    if dt == 0:
        dt = 1.0  # 避免除零

    vx = (last[0] - second_last[0]) / dt
    vy = (last[1] - second_last[1]) / dt
    vz = (last[2] - second_last[2]) / dt

    pred_points = []
    pred_times = []
    for i in range(1, num_points + 1):
        t = last_t + i * time_step
        pred_points.append([
            last[0] + vx * (t - last_t),
            last[1] + vy * (t - last_t),
            last[2] + vz * (t - last_t)
        ])
        pred_times.append(t)

    return pred_points, pred_times