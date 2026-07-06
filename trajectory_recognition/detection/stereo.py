"""
双目立体视觉 — Stereo Triangulation

利用双目标定参数，将左右目 2D 检测框三角测量为 3D 坐标。

原理:
  Z = (f_px * B) / d        — 深度 = 焦距 × 基线 / 视差
  X = (xl - cx) * Z / f_px  — 水平位移
  Y = (yl - cy) * Z / f_px  — 垂直位移
"""

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class StereoParams:
    """双目标定参数"""
    baseline: float = 1.0             # 基线距离（米）
    focal_length_px: float = 0.0      # 像素焦距（0 = 从 FOV 自动推算）
    fov_horizontal: float = 90.0      # 水平视场角（度），focal_length_px=0 时使用
    fov_vertical: float = 60.0        # 垂直视场角（度）
    resolution_width: int = 1920      # 图像宽度（像素）
    resolution_height: int = 1080     # 图像高度（像素）
    # 注：pitch/yaw/roll 旋转变换由 _to_world() 在相机坐标→世界坐标阶段统一处理，
    # 不在 StereoTriangulator 中使用。保留字段仅用于配置透传。
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0


class StereoTriangulator:
    """
    双目三角测量器。

    假设:
      - 两摄像头水平排列，光轴平行（或已知会聚角）
      - 已校正（行对齐），视差仅存在于水平方向
      - 相机坐标系: X 右, Y 下, Z 前（OpenCV 右手系，原点在双目光心中点）

    使用示例:
        params = StereoParams(baseline=1.0, fov_horizontal=90)
        stereo = StereoTriangulator(params)
        point_3d = stereo.triangulate(bbox_left, bbox_right)
    """

    # 可靠深度范围（米），用于视差合理性校验
    MIN_DEPTH = 0.5     # 太近不可靠（可能遮挡/模糊）
    MAX_DEPTH = 2000    # 硬上限，实际分辨率由视差决定

    def __init__(self, params: StereoParams):
        self.params = params

    # ── 像素焦距 ──

    @property
    def fx(self) -> float:
        """水平像素焦距: 优先使用 focal_length_px，否则从 FOV 推算"""
        if self.params.focal_length_px > 0:
            return self.params.focal_length_px
        return (self.params.resolution_width / 2) / math.tan(
            math.radians(self.params.fov_horizontal / 2)
        )

    @property
    def fy(self) -> float:
        """垂直像素焦距: fx 已指定则按正方形像素假设 fy≈fx，否则从 FOV 推算"""
        if self.params.focal_length_px > 0:
            return self.params.focal_length_px
        return (self.params.resolution_height / 2) / math.tan(
            math.radians(self.params.fov_vertical / 2)
        )

    @property
    def cx(self) -> float:
        """主点 X（假设图像中心）"""
        return self.params.resolution_width / 2.0

    @property
    def cy(self) -> float:
        """主点 Y（假设图像中心）"""
        return self.params.resolution_height / 2.0

    @property
    def _min_disparity(self) -> float:
        """最小可靠视差（像素），基于 MAX_DEPTH"""
        return (self.fx * self.params.baseline) / self.MAX_DEPTH

    @property
    def _max_disparity(self) -> float:
        """最大合理视差（像素），基于 MIN_DEPTH"""
        return (self.fx * self.params.baseline) / self.MIN_DEPTH

    # ── 三角测量 ──

    def triangulate(
        self,
        bbox_left: list[float],
        bbox_right: list[float],
    ) -> Optional[list[float]]:
        """
        对左右目同一目标的检测框进行三角测量。

        Args:
            bbox_left:  左目检测框 [x1, y1, x2, y2] 像素坐标
            bbox_right: 右目检测框 [x1, y1, x2, y2] 像素坐标

        Returns:
            3D 坐标 [x, y, z]（米），或 None（视差无效）
        """
        # 1. 取 bbox 中心点
        xl = (bbox_left[0] + bbox_left[2]) / 2.0
        yl = (bbox_left[1] + bbox_left[3]) / 2.0
        xr = (bbox_right[0] + bbox_right[2]) / 2.0
        yr = (bbox_right[1] + bbox_right[3]) / 2.0

        # 2. 视差（水平，已校正的双目只需水平视差）
        disparity = xl - xr
        min_disp = max(self._min_disparity, 1.0)  # 至少 1 像素
        if abs(disparity) < min_disp:
            return None  # 视差太小，无法可靠测距

        # 3. 深度 Z = f * B / d
        Z = (self.fx * self.params.baseline) / disparity

        # 距离合理性检查（使用动态计算的上限）
        max_z = min(self.MAX_DEPTH, (self.fx * self.params.baseline) / max(min_disp, 0.5))
        if Z <= self.MIN_DEPTH or Z > max_z:
            return None

        # 4. 相机坐标系（原点=双目光心中点, X=右 Y=↓ Z=前, OpenCV标准）
        X = (xl - self.cx) * Z / self.fx - self.params.baseline / 2
        # Y: 使用左右目 Y 坐标均值，减小垂直对齐误差
        Y = ((yl + yr) / 2.0 - self.cy) * Z / self.fy  # 图像坐标: y↓, 相机坐标: Y↓

        return [round(X, 3), round(Y, 3), round(Z, 3)]

    # ── 跨目匹配 ──

    def match_detections(
        self,
        dets_left: list,
        dets_right: list,
        max_y_diff: float = 50.0,
        max_size_ratio: float = 2.0,
        target_class_id: Optional[int] = None,
    ) -> list[tuple]:
        """
        匹配左右目的检测结果。

        匹配策略:
          1. 类别一致: class_id 必须相同（同一目标类型）
          2. 行对齐约束: |yl - yr| < max_y_diff（已校正的双目图像）
          3. 视差合理性: 视差在 [min_disp, max_disp] 范围内（深度约束）
          4. 尺寸约束: bbox 面积比不超过 max_size_ratio
          5. 水平约束: xl > xr（左目中的目标偏右，右目中的偏左）

        评分: 综合 y 对齐误差 + 尺寸偏差 + 类别权重

        Args:
            dets_left:  左目 Detection 列表
            dets_right: 右目 Detection 列表
            max_y_diff: 最大垂直像素差（越小编码越精确）
            max_size_ratio: bbox 面积比上限
            target_class_id: 目标类别 ID，若指定则额外加权优先匹配

        Returns:
            匹配对列表: [(det_left, det_right, [x, y, z]), ...]
        """
        matches = []
        used_right = set()

        for dl in dets_left:
            best_match = None
            best_score = float('inf')

            xl = (dl.bbox[0] + dl.bbox[2]) / 2
            yl = (dl.bbox[1] + dl.bbox[3]) / 2
            area_l = max((dl.bbox[2] - dl.bbox[0]) * (dl.bbox[3] - dl.bbox[1]), 1.0)

            for j, dr in enumerate(dets_right):
                if j in used_right:
                    continue

                # 类别一致性（核心约束）
                if dl.class_id != dr.class_id:
                    continue

                yr = (dr.bbox[1] + dr.bbox[3]) / 2
                xr = (dr.bbox[0] + dr.bbox[2]) / 2
                area_r = max((dr.bbox[2] - dr.bbox[0]) * (dr.bbox[3] - dr.bbox[1]), 1.0)

                # 行对齐
                y_diff = abs(yl - yr)
                if y_diff > max_y_diff:
                    continue

                # 水平约束（左目目标应在右目目标的右侧）
                disparity = xl - xr
                if disparity <= 0:
                    continue

                # 视差合理性（深度范围约束）
                if disparity < self._min_disparity or disparity > self._max_disparity:
                    continue

                # 尺寸约束
                area_ratio = max(area_l, area_r) / min(area_l, area_r)
                if area_ratio > max_size_ratio:
                    continue

                # 综合评分（越低越好）
                # y 对齐主导 + 尺寸偏差 + 视差偏离预期中值的惩罚
                expected_disp = (self._min_disparity + self._max_disparity) / 2
                disp_deviation = abs(disparity - expected_disp) / max(expected_disp, 1)
                size_penalty = (area_ratio - 1.0) * 50  # 面积比偏离 1.0 越多扣分越多

                score = y_diff + disp_deviation * 20 + size_penalty

                # 目标类别加权：优先匹配指定类别
                if target_class_id is not None:
                    if dl.class_id == target_class_id:
                        score *= 0.5  # 优先
                    else:
                        score *= 2.0  # 非目标类别降低优先级

                if score < best_score:
                    best_score = score
                    best_match = (j, dr)

            if best_match is not None:
                j, dr = best_match
                used_right.add(j)
                point_3d = self.triangulate(dl.bbox, dr.bbox)
                matches.append((dl, dr, point_3d))

        return matches

    # ── 工具 ──

    def estimate_distance(self, bbox_size_px: float, real_size_m: float = 0.3) -> float:
        """
        单目粗略测距（无双目时降级使用）。

        注意：基于目标在图像中的像素尺寸估算距离，假设目标正对摄像头、
        完整呈现其参考尺寸。若目标转向或部分遮挡，误差会显著增大。

          Z = (f * real_size) / bbox_size_px

        Args:
            bbox_size_px: bbox 宽度（像素）
            real_size_m: 目标真实尺寸（米），无人机约 0.3m

        Returns:
            估算距离（米）
        """
        if bbox_size_px <= 0:
            return float('inf')
        return (self.fx * real_size_m) / bbox_size_px
