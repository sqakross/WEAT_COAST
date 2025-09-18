from flask import Flask
from config import Config
import os, sys, io, logging
from logging.handlers import RotatingFileHandler
from extensions import db, login_manager
# + Sentry
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration
from sentry_sdk.integrations.logging import LoggingIntegration

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

# >>> LOG_LEVEL через env
_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
_level = getattr(logging, _level_name, logging.INFO)
root.setLevel(_level)

fh = RotatingFileHandler(os.path.join(LOG_DIR, 'app.log'),
                         maxBytes=2_000_000, backupCount=3, encoding='utf-8')
fh.setFormatter(formatter)

sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(formatter)

# Чтобы при debug-reloader не плодились хендлеры:
if not root.handlers:
    root.addHandler(fh)
    root.addHandler(sh)

# NEW: приглушим werkzeug
logging.getLogger("werkzeug").setLevel(logging.WARNING)

# --- 4) Flask app ---
# ВАЖНО: instance_relative_config=True даёт читать instance/config.py
app = Flask(__name__, instance_relative_config=True)
app.config.from_object(Config)
app.config.from_pyfile('config.py', silent=True)  # ← добавили

# NEW: ограничим размер аплоадов (32 МБ по умолчанию)
app.config.setdefault("MAX_CONTENT_LENGTH", 32 * 1024 * 1024)

# --- 4.1) Доп. пути для шаблонов (ДОБАВЛЕНО — после создания app) ---
from jinja2 import ChoiceLoader, FileSystemLoader
extra_templates = [
    os.path.join(Config.BASE_DIR, "inventory", "templates"),
    # добавляй доп. папки при необходимости
]
extra_loaders = [FileSystemLoader(p) for p in extra_templates if os.path.isdir(p)]
if extra_loaders:
    app.jinja_loader = ChoiceLoader([app.jinja_loader, *extra_loaders])
    try:
        searchpath = getattr(app.jinja_loader, "searchpath", None)
        logging.info("Jinja searchpath: %s", searchpath)
    except Exception:
        pass
else:
    logging.warning("No extra template dirs found among: %s", extra_templates)

# --- Sentry (включается если задан SENTRY_DSN в окружении) ---
SENTRY_DSN = os.getenv("SENTRY_DSN", "").strip()
if SENTRY_DSN:
    sentry_sdk.init(
        dsn=SENTRY_DSN,
        integrations=[
            FlaskIntegration(),
            LoggingIntegration(level=logging.INFO, event_level=logging.ERROR),
        ],
        traces_sample_rate=float(os.getenv("SENTRY_TRACES", "0.0")),
        profiles_sample_rate=float(os.getenv("SENTRY_PROFILES", "0.0")),
        send_default_pii=False,
        environment=os.getenv("ENVIRONMENT", "development"),
    )
    logging.info("Sentry initialized.")
else:
    logging.info("Sentry DSN not set; skipping Sentry init.")

# На всякий случай создадим фактическую instance-папку Flask
os.makedirs(app.instance_path, exist_ok=True)

db.init_app(app)
login_manager.init_app(app)
login_manager.login_view = 'auth.login'

# Проверим, что флаги подхватились (увидишь в логах)
logging.info(
    "WCCR flags: enabled=%s, dry=%s",
    app.config.get("WCCR_IMPORT_ENABLED"),
    app.config.get("WCCR_IMPORT_DRY_RUN"),
)

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

        # --- ensure new columns in SQLite (no Alembic needed) ---
        def _ensure_column(table: str, column: str, ddl_type: str):
            try:
                rows = db.session.execute(db.text(f"PRAGMA table_info({table})")).fetchall()
                names = {r[1] for r in rows}  # name is at index 1
                if column not in names:
                    logging.info(f"Adding column {column} to {table} ...")
                    db.session.execute(db.text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_type}"))
                    db.session.commit()
                    logging.info(f"Added column {column} to {table}.")
            except Exception as e:
                logging.exception(f"Failed to ensure column {table}.{column}: {e}")

        # наш новый столбец для группировки позиций по Appliance
        _ensure_column("work_order_parts", "unit_label", "TEXT")

    port = int(os.environ.get("PORT", 5000))

    # NEW: управляем debug/ssl через env без правки кода
    debug = os.getenv("DEBUG", "true").lower() == "true"
    use_ssl = os.getenv("USE_SSL", "1").lower() in ("1", "true", "yes")

    if use_ssl:
        logging.info(f"Starting dev server on https://0.0.0.0:{port} (adhoc TLS), debug={debug}")
        app.run(host='0.0.0.0', port=port, debug=debug, ssl_context='adhoc')
    else:
        logging.info(f"Starting dev server on http://0.0.0.0:{port}, debug={debug}")
        app.run(host='0.0.0.0', port=port, debug=debug)




