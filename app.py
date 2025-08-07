from flask import Flask
from config import Config
import os
# from app import app

# Import extensions
from extensions import db, login_manager

os.makedirs(os.path.join(Config.BASE_DIR, 'instance'), exist_ok=True)
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

# Create Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Initialize extensions
db.init_app(app)
login_manager.init_app(app)
login_manager.login_view = 'auth.login'

# Import and register Blueprints AFTER app is defined
from auth.routes import auth_bp
from inventory.routes import inventory_bp

app.register_blueprint(auth_bp)
app.register_blueprint(inventory_bp)

# Create DB tables before the first request
# @app.before_request
# def create_tables_once():
#     if not hasattr(app, 'tables_created'):
#         db.create_all()
#         app.tables_created = True

# Run
if __name__ == "__main__":
    with app.app_context():
        db.create_all()  # Создаем все таблицы, если их нет

    port = int(os.environ.get("PORT", 5000))
    # app.run(debug=True, port=port)
    app.run(host='0.0.0.0', port=5000, debug=True, ssl_context='adhoc')


