# app.py
from flask import Flask
from config import Config
import os, sys, io, logging
from logging.handlers import RotatingFileHandler

from extensions import db, login_manager

# --- 1) Форсируем UTF-8 для stdout/stderr, чтобы print с кириллицей не падал ---
def _force_utf8_stdio():
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        else:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        # Ничего страшного: просто не переопределили потоки
        pass

_force_utf8_stdio()

# --- 2) Директории (instance, uploads, logs) ---
os.makedirs(os.path.join(Config.BASE_DIR, 'instance'), exist_ok=True)
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
LOG_DIR = os.path.join(Config.BASE_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# --- 3) Логирование в файл + в консоль (UTF-8) ---
formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(name)s: %(message)s')
root = logging.getLogger()
root.setLevel(logging.INFO)

fh = RotatingFileHandler(os.path.join(LOG_DIR, 'app.log'),
                         maxBytes=2_000_000, backupCount=3, encoding='utf-8')
fh.setFormatter(formatter)

sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(formatter)

# Чтобы при debug-reloader не плодились хендлеры:
if not root.handlers:
    root.addHandler(fh)
    root.addHandler(sh)

# --- 4) Flask app ---
app = Flask(__name__)
app.config.from_object(Config)

db.init_app(app)
login_manager.init_app(app)
login_manager.login_view = 'auth.login'

from auth.routes import auth_bp
from inventory.routes import inventory_bp

app.register_blueprint(auth_bp)
app.register_blueprint(inventory_bp)

logging.info("Flask app configured and blueprints registered.")

# --- 5) Локальный запуск (dev, adhoc HTTPS) ---
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        logging.info("DB tables ensured (create_all).")

    port = int(os.environ.get("PORT", 5000))
    logging.info(f"Starting dev server on https://0.0.0.0:{port} (adhoc TLS)")
    app.run(host='0.0.0.0', port=port, debug=True, ssl_context='adhoc')

