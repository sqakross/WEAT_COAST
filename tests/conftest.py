# tests/conftest.py
import pytest
from flask import Flask
from config import Config
from extensions import db, login_manager

# импортируй блюпринты
from inventory.routes import inventory_bp        # если нужен
try:
    from orders.routes import orders_bp          # поправь путь под свой модуль
except Exception:
    orders_bp = None

class TestingConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"  # отдельная БД для тестов
    WTF_CSRF_ENABLED = False                        # чтобы POST без CSRF проходил
    LOGIN_DISABLED = True                           # отключить @login_required
    # при необходимости добавь SECRET_KEY и др.

@pytest.fixture
def app():
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(TestingConfig)

    # init extensions
    db.init_app(app)
    login_manager.init_app(app)

    # регистрируем блюпринты, если есть
    app.register_blueprint(inventory_bp)
    if orders_bp:
        app.register_blueprint(orders_bp)

    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()

@pytest.fixture
def client(app):
    return app.test_client()

@pytest.fixture
def auth(client):
    # Заглушка: LOGIN_DISABLED=True, но оставим интерфейс
    class Auth:
        def login(self): return True
    return Auth()
