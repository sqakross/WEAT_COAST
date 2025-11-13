from flask import Flask, request, redirect, url_for, send_file, abort
from config import Config
import os, sys, io, logging, time, ipaddress
from logging.handlers import RotatingFileHandler
from extensions import db, login_manager
from flask_login import current_user

# Sentry
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration
from sentry_sdk.integrations.logging import LoggingIntegration

from security import is_technician  # (если где-то ещё используется)

from datetime import datetime
from zoneinfo import ZoneInfo
from jinja2 import ChoiceLoader, FileSystemLoader

# -------------------------------------------------------------------
# 1) Форсируем UTF-8 для stdout/stderr (кириллица в print)
# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
# 2) Директории (instance, uploads, logs)
# -------------------------------------------------------------------
os.makedirs(os.path.join(Config.BASE_DIR, 'instance'), exist_ok=True)
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

LOG_DIR = os.path.join(Config.BASE_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# -------------------------------------------------------------------
# 3) Логирование в файл + в консоль (UTF-8)
# -------------------------------------------------------------------
formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(name)s: %(message)s')
root = logging.getLogger()
root.setLevel(logging.INFO)

logging.getLogger().setLevel(logging.INFO)
logging.getLogger("app").setLevel(logging.INFO)
logging.getLogger("inventory").setLevel(logging.INFO)
logging.getLogger("werkzeug").setLevel(logging.WARNING)

fh = RotatingFileHandler(
    os.path.join(LOG_DIR, 'app.log'),
    maxBytes=2_000_000,
    backupCount=3,
    encoding='utf-8'
)
fh.setFormatter(formatter)

sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(formatter)

if not root.handlers:
    root.addHandler(fh)
    root.addHandler(sh)

# -------------------------------------------------------------------
# 4) Flask app init
# -------------------------------------------------------------------
app = Flask(__name__, instance_relative_config=True)
app.config.from_object(Config)
app.config.from_pyfile('config.py', silent=True)  # instance/config.py (если есть)

# флаги импорта
app.config['WCCR_IMPORT_ENABLED'] = 1
app.config['WCCR_IMPORT_DRY_RUN'] = 0

# Ограничим размер аплоадов
app.config.setdefault("MAX_CONTENT_LENGTH", 32 * 1024 * 1024)

# Таймзона отображения времени
app.config.setdefault("DISPLAY_TZ", "America/Los_Angeles")

logging.info(
    "WCCR flags (FINAL): enabled=%s, dry=%s",
    app.config.get("WCCR_IMPORT_ENABLED"),
    app.config.get("WCCR_IMPORT_DRY_RUN"),
)

# -------------------------------------------------------------------
# 4.1) Доп. пути для шаблонов (inventory + supplier_returns)
# -------------------------------------------------------------------
extra_templates = [
    os.path.join(Config.BASE_DIR, "inventory", "templates"),
    os.path.join(Config.BASE_DIR, "supplier_returns", "templates"),
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

# -------------------------------------------------------------------
# 5) Sentry (если задан SENTRY_DSN)
# -------------------------------------------------------------------
SENTRY_DSN = os.getenv("SENTRY_DSN", "").strip()
if SENTRY_DSN:
    sentry_sdk.init(
        dsn=SENTRY_DSN,
        integrations=[FlaskIntegration(), LoggingIntegration(level=logging.INFO, event_level=logging.ERROR)],
        traces_sample_rate=float(os.getenv("SENTRY_TRACES", "0.0")),
        profiles_sample_rate=float(os.getenv("SENTRY_PROFILES", "0.0")),
        send_default_pii=False,
        environment=os.getenv("ENVIRONMENT", "development"),
    )
    logging.info("Sentry initialized.")
else:
    logging.info("Sentry DSN not set; skipping Sentry init.")

# -------------------------------------------------------------------
# 6) DB / Login manager
# -------------------------------------------------------------------
os.makedirs(app.instance_path, exist_ok=True)
db.init_app(app)
login_manager.init_app(app)
login_manager.login_view = 'auth.login'

logging.info(
    "WCCR flags: enabled=%s, dry=%s",
    app.config.get("WCCR_IMPORT_ENABLED"),
    app.config.get("WCCR_IMPORT_DRY_RUN"),
)

# -------------------------------------------------------------------
# 7) Jinja-фильтры локального времени
# -------------------------------------------------------------------
def _to_local(dt: datetime, fmt: str):
    if not dt:
        return "—"
    tzname = app.config.get("DISPLAY_TZ", "America/Los_Angeles")
    try:
        dt_utc = dt.replace(tzinfo=ZoneInfo("UTC"))
        dt_local = dt_utc.astimezone(ZoneInfo(tzname))
        return dt_local.strftime(fmt)
    except Exception:
        return dt.strftime(fmt)

@app.template_filter("local_dt")
def jinja_local_dt(dt: datetime, fmt: str = "%Y-%m-%d %H:%M"):
    return _to_local(dt, fmt)

@app.template_filter("local_date")
def jinja_local_date(dt: datetime, fmt: str = "%Y-%m-%d"):
    return _to_local(dt, fmt)

# -------------------------------------------------------------------
# 8) Blueprints
# -------------------------------------------------------------------
from auth.routes import auth_bp
from inventory.routes import inventory_bp

app.register_blueprint(auth_bp)
app.register_blueprint(inventory_bp)

# Supplier Returns — используем один blueprint из routes, префикс задаём здесь
from supplier_returns.routes import supplier_returns_bp
app.register_blueprint(supplier_returns_bp, url_prefix="/supplier_returns")

logging.info("Flask app configured and blueprints registered.")

# -------------------------------------------------------------------
# Раздача КОРНЕВОГО CA-корня клиентам (для установки доверия)
# -------------------------------------------------------------------
@app.get("/_download_ca")
def download_ca():
    """
    Скачивание корневого CA для установки на клиентских ПК.
    Ищи файл ssl/ca.crt (PEM) или ssl/wccr-root.cer (DER).
    Открой в браузере: https://<IP>:<PORT>/_download_ca
    """
    # Предпочтительно DER (для Windows). Если нет — отдадим PEM.
    der_path = os.path.join("ssl", "wccr-root.cer")
    pem_path = os.path.join("ssl", "ca.crt")

    if os.path.exists(der_path):
        send_path = der_path
        mtype = "application/x-x509-ca-cert"
        dname = "wccr-root.cer"
    elif os.path.exists(pem_path):
        send_path = pem_path
        mtype = "application/x-x509-ca-cert"
        dname = "wccr-root.cer"
    else:
        return "CA file not found. Put ssl/wccr-root.cer or ssl/ca.crt on server.", 404

    # (опционально) ограничим скачивание локальными/приватными адресами
    try:
        ip = ipaddress.ip_address(request.remote_addr or "127.0.0.1")
        if not (ip.is_private or ip.is_loopback):
            return abort(403)
    except Exception:
        pass

    return send_file(send_path, mimetype=mtype, as_attachment=True, download_name=dname, max_age=0)

# -------------------------------------------------------------------
# 9) Ограничение навигации для роли technician / viewer
# -------------------------------------------------------------------
ALLOWED_TECH_ENDPOINTS = {
    "inventory.wo_list",
    "inventory.wo_detail",
    "inventory.wo_confirm_lines",
    "inventory.issued_confirm_toggle",
    "inventory.reports_grouped",
    "inventory.view_invoice_pdf",
    "inventory.change_password",
    "auth.logout",
    "auth.login",
    "static",
}

ALLOWED_VIEWER_ENDPOINTS = {
    "inventory.wo_list",
    "inventory.wo_detail",
    "inventory.reports_grouped",
    "inventory.change_password",
    "auth.logout",
    "static",
}

@app.before_request
def restrict_role_routes():
    """
    Ограничиваем доступ по ролям 'technician' и 'viewer' *только* разрешёнными endpoint'ами.
    Для остальных ролей (admin/superadmin/user) — без ограничений.
    """
    try:
        if not getattr(current_user, "is_authenticated", False):
            return  # гость — пусть дойдёт до /login

        ep = (request.endpoint or "").strip()
        role = (getattr(current_user, "role", "") or "").strip().lower()

        if role == "technician":
            if ep in ALLOWED_TECH_ENDPOINTS or ep.endswith(".health") or ep.endswith(".ping"):
                return
            return redirect(url_for("inventory.wo_list"))

        if role == "viewer":
            if ep in ALLOWED_VIEWER_ENDPOINTS or ep.endswith(".health") or ep.endswith(".ping"):
                return
            return redirect(url_for("inventory.wo_list"))

        # admin/superadmin/user — не ограничиваем
        return
    except Exception:
        try:
            return redirect(url_for("inventory.wo_list"))
        except Exception:
            return

# -------------------------------------------------------------------
# 10) Вспомогательные миграции (безопасные автополевые апдейты)
# -------------------------------------------------------------------
def _ensure_column(table: str, column: str, ddl_type: str):
    try:
        row = db.session.execute(
            db.text("SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name=:t"),
            {"t": table}
        ).fetchone()
        if not row:
            logging.info("Skip ensure column %s.%s: table does not exist", table, column)
            return

        rows = db.session.execute(db.text(f"PRAGMA table_info({table})")).fetchall()
        names = {r[1] for r in rows}
        if column in names:
            return

        logging.info("Adding column %s to %s ...", column, table)
        db.session.execute(db.text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_type}"))
        db.session.commit()
        logging.info("Added column %s to %s.", column, table)
    except Exception as e:
        logging.exception("Failed to ensure column %s.%s: %s", table, column, e)

def _backfill_units_for_legacy_parts():
    from models import WorkOrder, WorkUnit
    orders = WorkOrder.query.all()
    created_units = 0

    for wo in orders:
        if not wo.units:
            unit = WorkUnit(
                work_order_id=wo.id,
                brand=getattr(wo, "brand", "") or "",
                model=getattr(wo, "model", "") or "",
                serial=getattr(wo, "serial", "") or "",
            )
            db.session.add(unit)
            db.session.flush()
            created_units += 1

    res = db.session.execute(db.text("""
        UPDATE work_order_parts
        SET unit_id = (
            SELECT wu.id
            FROM work_units wu
            WHERE wu.work_order_id = work_order_parts.work_order_id
            ORDER BY wu.id ASC
            LIMIT 1
        )
        WHERE unit_id IS NULL AND work_order_id IS NOT NULL
    """))
    db.session.commit()
    logging.info("Backfill: created %s units; updated %s parts", created_units, res.rowcount)

def _boot_ensure_core_columns():
    try:
        with app.app_context():
            _ensure_column("work_order_parts", "ordered_flag", "INTEGER DEFAULT 0")
            _ensure_column("work_order_parts", "ordered_date", "DATE")
            _ensure_column("work_orders", "ordered_date", "DATE")
    except Exception:
        logging.exception("Boot ensure columns failed")

def _ensure_consumption_columns():
    try:
        with app.app_context():
            _ensure_column("issued_part_record", "consumed_qty",  "INTEGER")
            _ensure_column("issued_part_record", "consumed_flag", "INTEGER DEFAULT 0")
            _ensure_column("issued_part_record", "consumed_at",   "DATETIME")
            _ensure_column("issued_part_record", "consumed_by",   "TEXT")
            _ensure_column("issued_part_record", "consumed_note", "TEXT")

            _ensure_column("issued_batch", "consumed_flag", "INTEGER DEFAULT 0")
            _ensure_column("issued_batch", "consumed_at",   "DATETIME")
            _ensure_column("issued_batch", "consumed_by",   "TEXT")
            _ensure_column("issued_batch", "is_stock", "INTEGER DEFAULT 0")
    except Exception:
        logging.exception("Ensure consumption columns failed")

_boot_ensure_core_columns()
_ensure_consumption_columns()

# -------------------------------------------------------------------
# 11) Чистка старых .docx из uploads
# -------------------------------------------------------------------
def cleanup_old_reports(folder: str, days_old: int = 3):
    try:
        now_ts = time.time()
        max_age_seconds = days_old * 24 * 60 * 60

        for name in os.listdir(folder):
            if not name.lower().endswith(".docx"):
                continue
            full_path = os.path.join(folder, name)
            if not os.path.isfile(full_path):
                continue
            try:
                stat = os.stat(full_path)
                age_seconds = now_ts - stat.st_mtime
                if age_seconds > max_age_seconds:
                    os.remove(full_path)
                    logging.info("CLEANUP: removed old report %s", full_path)
            except Exception as e:
                logging.warning("CLEANUP: failed to remove %s: %s", full_path, e)
    except Exception as e:
        logging.warning("CLEANUP: general failure: %s", e)

# -------------------------------------------------------------------
# 12) Локальный запуск (dev, HTTPS с локальным сертификатом)
# -------------------------------------------------------------------
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        logging.info("DB tables ensured (create_all).")

        _ensure_column("work_order_parts", "unit_label", "TEXT")
        _ensure_column("work_order_parts", "unit_id", "INTEGER")
        _ensure_column("work_order_parts", "unit_cost", "REAL")

        try:
            _backfill_units_for_legacy_parts()
        except Exception:
            logging.exception("Backfill failed")

        cleanup_old_reports(app.config["UPLOAD_FOLDER"], days_old=3)
        logging.info("Old .docx cleanup complete.")

    port = int(os.environ.get("PORT", 5000))
    debug = os.getenv("DEBUG", "true").lower() == "true"
    use_ssl = os.getenv("USE_SSL", "1").lower() in ("1", "true", "yes")

    if use_ssl:
        # ВАЖНО: используем серверный сертификат, подписанный нашим CA
        cert_path = os.path.join("ssl", "server.crt")
        key_path  = os.path.join("ssl", "server.key")

        if os.path.exists(cert_path) and os.path.exists(key_path):
            logging.info(f"Starting secure server on https://0.0.0.0:{port}, debug={debug}")
            app.run(host="0.0.0.0", port=port, debug=debug, ssl_context=(cert_path, key_path))
        else:
            logging.warning("SSL server cert not found (ssl/server.crt|server.key). Falling back to HTTP.")
            app.run(host="0.0.0.0", port=port, debug=debug)
    else:
        logging.info(f"Starting HTTP server on http://0.0.0.0:{port}, debug={debug}")
        app.run(host="0.0.0.0", port=port, debug=debug)
