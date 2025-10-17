from flask import Flask, request, redirect, url_for
from config import Config
import os, sys, io, logging
from logging.handlers import RotatingFileHandler
from extensions import db, login_manager
from flask_login import current_user
from sqlalchemy.orm import relationship


# + Sentry
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration
from sentry_sdk.integrations.logging import LoggingIntegration
from security import is_technician

# --- 1) Форсируем UTF-8 для stdout/stderr (кириллица в print) ---
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

_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
_level = getattr(logging, _level_name, logging.INFO)
root.setLevel(_level)

fh = RotatingFileHandler(os.path.join(LOG_DIR, 'app.log'),
                         maxBytes=2_000_000, backupCount=3, encoding='utf-8')
fh.setFormatter(formatter)

sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(formatter)

if not root.handlers:
    root.addHandler(fh)
    root.addHandler(sh)

logging.getLogger("werkzeug").setLevel(logging.WARNING)

# --- 4) Flask app ---
app = Flask(__name__, instance_relative_config=True)
app.config.from_object(Config)
app.config.from_pyfile('config.py', silent=True)  # instance/config.py (если есть)

app.config['WCCR_IMPORT_ENABLED'] = 1   # применяем импорт
app.config['WCCR_IMPORT_DRY_RUN'] = 0   # не dry-run

logging.info(
    "WCCR flags (FINAL): enabled=%s, dry=%s",
    app.config.get("WCCR_IMPORT_ENABLED"),
    app.config.get("WCCR_IMPORT_DRY_RUN"),
)

# Ограничим размер аплоадов (32 МБ по умолчанию)
app.config.setdefault("MAX_CONTENT_LENGTH", 32 * 1024 * 1024)

# --- added: таймзона для отображения времени ---
app.config.setdefault("DISPLAY_TZ", "America/Los_Angeles")

# --- 4.1) Доп. пути для шаблонов ---
from jinja2 import ChoiceLoader, FileSystemLoader
extra_templates = [
    os.path.join(Config.BASE_DIR, "inventory", "templates"),
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

# --- Sentry (включается если задан SENTRY_DSN) ---
SENTRY_DSN = os.getenv("SENTRY_DSN", "").strip()
if SENTRY_DSN:
    sentry_sdk.init(
        dsn=SENTRY_DSN,
        integrations=[FlaskIntegration(),
                      LoggingIntegration(level=logging.INFO, event_level=logging.ERROR)],
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

# Проверим флаги
logging.info(
    "WCCR flags: enabled=%s, dry=%s",
    app.config.get("WCCR_IMPORT_ENABLED"),
    app.config.get("WCCR_IMPORT_DRY_RUN"),
)

# --- added: Jinja-фильтры локального времени ---
from datetime import datetime
from zoneinfo import ZoneInfo

def _to_local(dt: datetime, fmt: str):
    if not dt:
        return "—"
    tzname = app.config.get("DISPLAY_TZ", "America/Los_Angeles")
    try:
        # считаем, что в БД время хранится как UTC-naive
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

from auth.routes import auth_bp
from inventory.routes import inventory_bp
app.register_blueprint(auth_bp)
app.register_blueprint(inventory_bp)

logging.info("Flask app configured and blueprints registered.")

# --- Ограничение навигации для роли technician (white-list) ---
# Разрешённые endpoint'ы для техника. Подставь/оставь только те, что реально есть.
ALLOWED_TECH_ENDPOINTS = {
    # Work Orders
    "inventory.wo_list",
    "inventory.wo_detail",

    # Если есть подтверждения и печать:
    "inventory.wo_confirm_lines",   # <-- твой POST-роут подтверждения
    # "inventory.batch_confirm",    # если нет — можно снять
    # "inventory.print_report",     # если нет — можно снять

    # Отчёты — только группированный, но дальше мы подстрижём параметры
    "inventory.reports_grouped",

    # Смена пароля и выход
    "inventory.change_password",
    "auth.logout",

    # login и статика
    "auth.login",
    "static",
}

# VIEWER: только Work Orders (лист + деталка) и Reports; смена пароля себе; выход; статика.
ALLOWED_VIEWER_ENDPOINTS = {
    "inventory.wo_list",
    "inventory.wo_detail",
    "inventory.reports_grouped",

    "inventory.change_password",
    "auth.logout",

    "static",
}

def _is_allowed_for_tech(endpoint: str) -> bool:
    if not endpoint:
        return False

    # Статика всегда нужна
    if endpoint == "static":
        return True

    # Work Orders (список, детали, подтверждение)
    if endpoint.startswith("inventory.wo_"):
        # включает: inventory.wo_list, inventory.wo_detail, inventory.wo_confirm_lines, и т.д.
        return True

    # Отчёты технику по его работам (кнопка Open Report ведёт сюда)
    if endpoint.startswith("inventory.reports_"):
        # например: inventory.reports_grouped
        return True

    # Печать PDF (кнопка Print)
    if endpoint in {"inventory.invoice_pdf", "inventory.print_invoice", "inventory.print_report"}:
        return True

    # Смена пароля и выход
    if endpoint in {"inventory.change_password", "auth.logout"}:
        return True

    return False

from urllib.parse import urlencode

@app.before_request
def restrict_role_routes():
    """
    Ограничиваем доступ по ролям 'technician' и 'viewer' *только* разрешёнными endpoint'ами.
    Все остальные запросы уводим на список Work Orders.
    Для остальных ролей (admin/superadmin/user) — без ограничений (остаются проверки в роутерах/шаблонах).
    """
    try:
        if not getattr(current_user, "is_authenticated", False):
            return  # гость — пусть доходит до /login

        ep = (request.endpoint or "").strip()
        role = (getattr(current_user, "role", "") or "").strip().lower()

        if role == "technician":
            if ep in ALLOWED_TECH_ENDPOINTS:
                return
            # health/ping допускаем на всякий случай
            if ep.endswith(".health") or ep.endswith(".ping"):
                return
            return redirect(url_for("inventory.wo_list"))

        if role == "viewer":
            if ep in ALLOWED_VIEWER_ENDPOINTS:
                return
            if ep.endswith(".health") or ep.endswith(".ping"):
                return
            return redirect(url_for("inventory.wo_list"))

        # другие роли — без общего редиректа (остаются локальные проверки прав)
        return
    except Exception:
        # в случае ошибки не блокируем, но для safety уводим на список WO
        try:
            return redirect(url_for("inventory.wo_list"))
        except Exception:
            return
# ----------------------------
# ВСПОМОГАТЕЛЬНЫЕ МИГРАЦИИ
# ----------------------------
def _ensure_column(table: str, column: str, ddl_type: str):
    """Безопасно добавляет колонку, если её нет (SQLite).
       Если таблицы нет — пропускаем без ошибки."""
    try:
        # есть ли таблица?
        row = db.session.execute(
            db.text("SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name=:t"),
            {"t": table}
        ).fetchone()
        if not row:
            logging.info("Skip ensure column %s.%s: table does not exist", table, column)
            return

        # есть ли колонка?
        rows = db.session.execute(db.text(f"PRAGMA table_info({table})")).fetchall()
        names = {r[1] for r in rows}  # name на индексе 1
        if column in names:
            return

        logging.info("Adding column %s to %s ...", column, table)
        db.session.execute(db.text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_type}"))
        db.session.commit()
        logging.info("Added column %s to %s.", column, table)
    except Exception as e:
        logging.exception("Failed to ensure column %s.%s: %s", table, column, e)

def _backfill_units_for_legacy_parts():
    """
    Разово мигрирует старые данные:
    - создаёт WorkUnit для WorkOrder, если его нет;
    - прописывает unit_id в work_order_parts, где он NULL, по work_order_id.
    Повторный запуск безопасен.
    """
    from models import WorkOrder, WorkUnit

    # 1) создать недостающие units
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

    # 2) привязать части без unit_id к первому unit заказа
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

# === ГАРАНТИЯ КОЛОНОК ПРИ ЛЮБОМ СТАРТЕ (включая flask CLI) ===
def _boot_ensure_core_columns():
    try:
        with app.app_context():
            # строки заказов
            _ensure_column("work_order_parts", "ordered_flag", "INTEGER DEFAULT 0")
            _ensure_column("work_order_parts", "ordered_date", "DATE")
            # сами заказы
            _ensure_column("work_orders", "ordered_date", "DATE")
    except Exception:
        logging.exception("Boot ensure columns failed")

_boot_ensure_core_columns()
# === конец boot ensure ===

# --- 5) Локальный запуск (dev, adhoc HTTPS) ---
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        logging.info("DB tables ensured (create_all).")

        # существовавший столбец
        _ensure_column("work_order_parts", "unit_label", "TEXT")
        # НОВОЕ: ключ к WorkUnit (для новой схемы)
        _ensure_column("work_order_parts", "unit_id", "INTEGER")
        # НОВОЕ: цена строки
        _ensure_column("work_order_parts", "unit_cost", "REAL")

        # Разовая миграция старых данных → unit_id
        try:
            _backfill_units_for_legacy_parts()
        except Exception:
            logging.exception("Backfill failed")

    port = int(os.environ.get("PORT", 5000))
    debug = os.getenv("DEBUG", "true").lower() == "true"
    use_ssl = os.getenv("USE_SSL", "1").lower() in ("1", "true", "yes")

    logging.info(f"Starting dev server on https://0.0.0.0:{port} (adhoc TLS), debug={debug}")
    app.run(host='0.0.0.0', port=port, debug=debug, ssl_context='adhoc')
    # else:
    #     logging.info(f"Starting dev server on http://0.0.0.0:{port}, debug={debug}")
    #     app.run(host='0.0.0.0', port=port, debug=debug)


