from flask import Flask
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager
from datetime import timedelta
import os

db = SQLAlchemy()
jwt = JWTManager()

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'ckd-nephroai-secret-key-2024-secure'
    app.config['JWT_SECRET_KEY'] = 'ckd-nephroai-jwt-2024-secure-longkey'
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ckd_platform.db')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    db.init_app(app)
    jwt.init_app(app)
    with app.app_context():
        from models.user import User
        from models.prediction import Prediction
        from routes.auth import auth_bp
        from routes.predict import predict_bp
        from routes.admin import admin_bp
        from routes.analytics import analytics_bp
        app.register_blueprint(auth_bp, url_prefix='/api/auth')
        app.register_blueprint(predict_bp, url_prefix='/api')
        app.register_blueprint(admin_bp, url_prefix='/api/admin')
        app.register_blueprint(analytics_bp, url_prefix='/api/analytics')
        db.create_all()
        from werkzeug.security import generate_password_hash
        if not User.query.filter_by(email='admin@ckd.ai').first():
            db.session.add(User(username='Admin', email='admin@ckd.ai', password_hash=generate_password_hash('Admin@123'), role='admin'))
            db.session.commit()
            print("✅ Admin created")
        else:
            print("✅ Admin ready")
    return app

if __name__ == '__main__':
    create_app().run(debug=False, port=5000, host='0.0.0.0')
