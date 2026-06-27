"""
路由注册 — 统一注册所有蓝图 + 全局错误处理
"""
from flask import Flask, jsonify


def register_blueprints(app: Flask):
    from modules.routes.main import main_bp
    from modules.routes.api import api_bp
    from modules.routes.api_predict import predict_bp
    from modules.routes.analysis import analysis_bp
    from modules.routes.api_report import report_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp)
    app.register_blueprint(predict_bp)
    app.register_blueprint(report_bp)
    app.register_blueprint(analysis_bp, url_prefix='/analysis')


def register_error_handlers(app: Flask):
    @app.errorhandler(400)
    def bad_request(e):
        return jsonify({'success': False, 'error': str(e.description)}), 400

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({'success': False, 'error': '接口不存在'}), 404

    @app.errorhandler(500)
    def server_error(e):
        return jsonify({'success': False, 'error': '服务器内部错误'}), 500
