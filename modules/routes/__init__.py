"""
路由包 — 注册所有蓝图，统一错误处理
"""
from flask import Flask, jsonify


def register_blueprints(app: Flask):
    """向 Flask 应用注册所有蓝图"""
    from modules.routes.main import main_bp
    from modules.routes.api_data import data_bp
    from modules.routes.api_predict import predict_bp
    from modules.routes.api_backup import backup_bp
    from modules.routes.analysis import analysis_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(data_bp)
    app.register_blueprint(predict_bp)
    app.register_blueprint(backup_bp)
    app.register_blueprint(analysis_bp, url_prefix='/analysis')


def register_error_handlers(app: Flask):
    """注册全局错误处理"""

    @app.errorhandler(400)
    def bad_request(e):
        return jsonify({'success': False, 'error': str(e.description)}), 400

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({'success': False, 'error': '接口不存在'}), 404

    @app.errorhandler(405)
    def method_not_allowed(e):
        return jsonify({'success': False, 'error': '请求方法不允许'}), 405

    @app.errorhandler(500)
    def server_error(e):
        return jsonify({'success': False, 'error': '服务器内部错误'}), 500
