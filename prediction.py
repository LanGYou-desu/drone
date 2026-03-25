def generate_prediction(points, num_points=5):
    if len(points) < 2:
        return []
    last = points[-1]
    second_last = points[-2]
    dx = last[0] - second_last[0]
    dy = last[1] - second_last[1]
    dz = last[2] - second_last[2]
    pred = []
    for i in range(1, num_points + 1):
        pred.append([
            last[0] + dx * i,
            last[1] + dy * i,
            last[2] + dz * i
        ])
    return pred