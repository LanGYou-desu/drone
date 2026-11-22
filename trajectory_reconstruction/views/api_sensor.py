"""
多传感器协同 API — 时间同步 / 三维重建 / 智能评估

路由:
  GET  /api/sensors/status        当前传感器概览
  POST /api/sensors/reconstruct   执行 同步 -> 融合 -> 重建
  POST /api/sensors/assess        执行 重建 -> 预测 -> 风险评估
"""
from flask import Blueprint, jsonify, request

from trajectory_reconstruction.services import sensor_service

sensor_bp = Blueprint('sensors', __name__)


@sensor_bp.route('/api/sensors/status', methods=['GET'])
def sensors_status():
    """返回运行时各平台轨迹概览。"""
    try:
        return jsonify({'success': True, 'sensors': sensor_service.sensor_status()})
    except Exception as exc:   # noqa: BLE001
        return jsonify({'success': False, 'error': str(exc)}), 500


@sensor_bp.route('/api/sensors/reconstruct', methods=['POST'])
def sensors_reconstruct():
    """时间同步 + 多传感器融合 + 三维重建。"""
    payload = request.get_json(silent=True) or {}
    try:
        result = sensor_service.run_pipeline(payload)
        return jsonify(result)
    except ValueError as exc:
        return jsonify({'success': False, 'error': str(exc)}), 400
    except Exception as exc:   # noqa: BLE001
        return jsonify({'success': False, 'error': f'重建失败: {exc}'}), 500


@sensor_bp.route('/api/sensors/assess', methods=['POST'])
def sensors_assess():
    """智能风险评估（含 KinFlow 未来预测）。"""
    payload = request.get_json(silent=True) or {}
    try:
        result = sensor_service.assess_scenario(payload)
        if result.get('success'):
            # 防护区/预测视界持久化，刷新后评估参数与 3D 防护区保持一致
            sensor_service.update_collab_params({
                'zone': payload.get('zone'),
                'prediction_horizon_s': payload.get('prediction_horizon_s'),
            })
        return jsonify(result)
    except ValueError as exc:
        return jsonify({'success': False, 'error': str(exc)}), 400
    except Exception as exc:   # noqa: BLE001
        return jsonify({'success': False, 'error': f'评估失败: {exc}'}), 500
