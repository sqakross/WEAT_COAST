from flask import Flask, request, redirect, url_for
from config import Config
import os, sys, io, logging, time
from logging.handlers import RotatingFileHandler
from extensions import db, login_manager
from flask_login import current_user
from sqlalchemy.orm import relationship

# Sentry
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration
from sentry_sdk.integrations.logging import LoggingIntegration

from security import is_technician  # (оставляем импорт если где-то ещё юзается)

from datetime import datetime
from zoneinfo import ZoneInfo
from jinja2 import ChoiceLoader, FileSystemLoader
from urllib.parse import urlencode


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
#    ВАЖНО: жёстко ограничиваемся уровнем INFO, никакого DEBUG спама
# -------------------------------------------------------------------
formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(name)s: %(message)s')

root = logging.getLogger()
root.setLevel(logging.INFO)

# на всякий случай пробиваем для типичных логгеров нашего проекта
logging.getLogger().setLevel(logging.INFO)
logging.getLogger("app").setLevel(logging.INFO)
logging.getLogger("inventory").setLevel(logging.INFO)

# Flask dev server (werkzeug) оставляем повыше, чтобы не лил каждую строчку запроса
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
app.config['WCCR_IMPORT_ENABLED'] = 1   # применяем импорт
app.config['WCCR_IMPORT_DRY_RUN'] = 0   # не dry-run

# Ограничим размер аплоадов (32 МБ по умолчанию)
app.config.setdefault("MAX_CONTENT_LENGTH", 32 * 1024 * 1024)

# Таймзона отображения времени
app.config.setdefault("DISPLAY_TZ", "America/Los_Angeles")

logging.info(
    "WCCR flags (FINAL): enabled=%s, dry=%s",
    app.config.get("WCCR_IMPORT_ENABLED"),
    app.config.get("WCCR_IMPORT_DRY_RUN"),
)


# -------------------------------------------------------------------
# 4.1) Доп. пути для шаблонов
# -------------------------------------------------------------------
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


# -------------------------------------------------------------------
# 5) Sentry (если задан SENTRY_DSN)
# -------------------------------------------------------------------
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


# -------------------------------------------------------------------
# 6) DB / Login manager
# -------------------------------------------------------------------
# На всякий случай создадим фактическую instance-папку Flask
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


# -------------------------------------------------------------------
# 8) Blueprints
# -------------------------------------------------------------------
from auth.routes import auth_bp
from inventory.routes import inventory_bp

app.register_blueprint(auth_bp)
app.register_blueprint(inventory_bp)

logging.info("Flask app configured and blueprints registered.")


# -------------------------------------------------------------------
# 9) Ограничение навигации для роли technician / viewer
# -------------------------------------------------------------------
ALLOWED_TECH_ENDPOINTS = {
    # Work Orders
    "inventory.wo_list",
    "inventory.wo_detail",

    # подтверждения / чекбоксы
    "inventory.wo_confirm_lines",
    "inventory.issued_confirm_toggle",

    # отчёты / документы
    "inventory.reports_grouped",

    # >>> НОВОЕ: печать PDF инвойса <<<
    "inventory.view_invoice_pdf",

    # профиль
    "inventory.change_password",
    "auth.logout",

    # login и статика
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

def _is_allowed_for_tech(endpoint: str) -> bool:
    if not endpoint:
        return False

    # статика
    if endpoint == "static":
        return True

    # любые Work Order эндпоинты (список, детали, confirm и т.д.)
    if endpoint.startswith("inventory.wo_"):
        return True

    # отчёты (grouped и т.п.)
    if endpoint.startswith("inventory.reports_"):
        return True

    # >>> НОВОЕ: явно разрешаем печать инвойса для техника <<<
    if endpoint in {
        "inventory.view_invoice_pdf",  # наш основной PDF
        "inventory.invoice_pdf",       # если у тебя было старое имя
        "inventory.print_invoice",
        "inventory.print_report",
    }:
        return True

    # профиль / выход
    if endpoint in {"inventory.change_password", "auth.logout", "auth.login"}:
        return True

    return False

@app.before_request
def restrict_role_routes():
    """
    Ограничиваем доступ по ролям 'technician' и 'viewer' *только* разрешёнными endpoint'ами.
    Все остальные запросы уводим на список Work Orders.
    Для остальных ролей (admin/superadmin/user) — без ограничений.
    """
    try:
        if not getattr(current_user, "is_authenticated", False):
            return  # гость — пусть дойдёт до /login

        ep = (request.endpoint or "").strip()
        role = (getattr(current_user, "role", "") or "").strip().lower()

        if role == "technician":
            if ep in ALLOWED_TECH_ENDPOINTS:
                return
            if ep.endswith(".health") or ep.endswith(".ping"):
                return
            return redirect(url_for("inventory.wo_list"))

        if role == "viewer":
            if ep in ALLOWED_VIEWER_ENDPOINTS:
                return
            if ep.endswith(".health") or ep.endswith(".ping"):
                return
            return redirect(url_for("inventory.wo_list"))

        # admin/superadmin/user — не ограничиваем тут
        return
    except Exception:
        # safety fallback
        try:
            return redirect(url_for("inventory.wo_list"))
        except Exception:
            return


# -------------------------------------------------------------------
# 10) Вспомогательные миграции (безопасные автополевые апдейты)
# -------------------------------------------------------------------
def _ensure_column(table: str, column: str, ddl_type: str):
    """
    Безопасно добавляет колонку, если её нет (SQLite).
    Если таблицы нет — пропускаем без ошибки.
    """
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
        db.session.execute(
            db.text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_type}")
        )
        db.session.commit()
        logging.info("Added column %s to %s.", column, table)
    except Exception as e:
        logging.exception("Failed to ensure column %s.%s: %s", table, column, e)


def _backfill_units_for_legacy_parts():
    """
    Разово мигрирует старые данные:
    - создаёт WorkUnit для WorkOrder, если его нет;
    - прописывает unit_id в work_order_parts, где он NULL.
    Повторный запуск безопасен.
    """
    from models import WorkOrder, WorkUnit

    orders = WorkOrder.query.all()
    created_units = 0

    # 1) создать недостающие units
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


def _boot_ensure_core_columns():
    """
    Гарантируем критичные колонки, чтобы не падало при старте.
    """
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


# -------------------------------------------------------------------
# 11) Чистка старых .docx из uploads (старше N дней)
# -------------------------------------------------------------------
def cleanup_old_reports(folder: str, days_old: int = 3):
    """
    Удаляет .docx файлы из `folder`, которым больше days_old дней.
    Если файл залочен Word'ом — просто логируем warning и идём дальше.
    """
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
# 12) Локальный запуск (dev, adhoc HTTPS)
# -------------------------------------------------------------------
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        logging.info("DB tables ensured (create_all).")

        # существовавший столбец
        _ensure_column("work_order_parts", "unit_label", "TEXT")
        # связь с WorkUnit
        _ensure_column("work_order_parts", "unit_id", "INTEGER")
        # цена строки
        _ensure_column("work_order_parts", "unit_cost", "REAL")

        # Разовая миграция старых данных → unit_id
        try:
            _backfill_units_for_legacy_parts()
        except Exception:
            logging.exception("Backfill failed")

        # авто-чистка старых (ненужных) DOCX отчётов
        cleanup_old_reports(app.config["UPLOAD_FOLDER"], days_old=3)
        logging.info("Old .docx cleanup complete.")

    port = int(os.environ.get("PORT", 5000))
    debug = os.getenv("DEBUG", "true").lower() == "true"
    use_ssl = os.getenv("USE_SSL", "1").lower() in ("1", "true", "yes")

    logging.info(
        f"Starting dev server on https://0.0.0.0:{port} (adhoc TLS), debug={debug}"
    )
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug,
        ssl_context='adhoc'
    )
    # Если без TLS нужно будет:
    # logging.info(f"Starting dev server on http://0.0.0.0:{port}, debug={debug}")
    # app.run(host='0.0.0.0', port=port, debug=debug)

