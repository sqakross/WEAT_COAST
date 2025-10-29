from __future__ import annotations
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import func, or_, and_
from models import IssuedPartRecord, WorkOrder, WorkOrderPart, TechReceiveLog, IssuedBatch, Part, ReceivingBatch, \
    ReceivingItem, OrderItem
from services.receiving import unpost_receiving_batch
import json
from services.receiving import post_receiving_batch
from services.receiving_import import create_receiving_from_rows
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
from flask import (
    Blueprint, render_template, request, redirect, url_for,
    flash, send_file, jsonify, after_this_request,
    current_app,abort, session,                   # NEW
)
from markupsafe import Markup
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from werkzeug.exceptions import abort
from werkzeug.security import generate_password_hash, check_password_hash
# from sqlalchemy import or_
from urllib.parse import urlencode
import re,sqlite3,os

import pandas as pd
from io import BytesIO
from datetime import datetime, timedelta, time, date
from collections import defaultdict

from config import Config
from extensions import db
# from models import Part, IssuedPartRecord, User
from utils.invoice_generator import generate_invoice_pdf
# from models.order_items import OrderItem
from models import User, ROLE_SUPERADMIN, ROLE_ADMIN, ROLE_USER, ROLE_VIEWER,ROLE_TECHNICIAN, Part, WorkOrder, WorkOrderPart
from sqlalchemy import or_
from pathlib import Path
import logging

# PDF (ReportLab)
from reportlab.lib.pagesizes import letter, landscape
# from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
# from reportlab.lib import colors
# from reportlab.lib.styles import getSampleStyleSheet

# Сравнение корзин / экспорт
from compare_cart.run_compare import get_marcone_items, check_cart_items, export_to_docx
from compare_cart.run_compare_reliable import get_reliable_items

# Импорт на склад (наш новый функционал)
from .import_rules import load_table, normalize_table, build_receive_movements
from .import_ledger import has_key, add_key
                              # NEW
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import LETTER
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

inventory_bp = Blueprint('inventory', __name__)
EPORT_PATH = os.path.join(os.path.dirname(__file__), '..', 'marcone_inventory_report.docx')

DB_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "instance", "inventory.db"))

_MARKER_RX = re.compile(r'^\s*shipment\s*mark(?:ing)?\s*[:\-]?', re.I)


_PREFIX_RE = re.compile(r'^\s*(wo|job|pn|model|brand|serial)\s*:\s*(.+)\s*$', re.I)
_PN_TOKEN_RX = re.compile(r'^([A-Za-z0-9][A-Za-z0-9\-\/\._]*)\s+(.+)$')

_units_re = re.compile(r"^units\[(\d+)\]\[(brand|model|serial)\]$")
_rows_re  = re.compile(
    r"^units\[(\d+)\]\[rows\]\[(\d+)\]\[(part_number|part_name|quantity|alt_numbers|supplier|backorder_flag|line_status|unit_cost|location)\]$")
_rows_flat_re = re.compile(
    r"^rows\[(\d+)\]\[(part_number|part_name|quantity|alt_numbers|supplier|backorder_flag|line_status|unit_cost|location)\]$"
)

_PN_FIRST_TOKEN_RX = re.compile(r'^([A-Za-z0-9][A-Za-z0-9\-\/\._]*)\s+(.+)$')
_PN_SPLIT_RX  = re.compile(r'^([A-Za-z0-9][A-Za-z0-9\-\/\._]*)\s+(.+)$')
_TOTALS_RX    = re.compile(r'^\s*(order\s*total|orderinetot|ordertot|subtotal|total)\b', re.I)


# --- Invoice numbering baseline ---
INVOICE_START_AT = 140  # новые инвойсы начнутся с 000140

ALLOWED_PDF = {".pdf"}

# Ключ → дефолтная локация
SUPPLIER_LOC_DEFAULTS = {
    "reliable": "REL",
    "marcone":  "MAR",
    "marcon":   "MAR",
}

# inventory/routes.py  (добавь рядом с _create_batch_for_records)
# --- TECH ROLE HELPER (fallback, если нет security.py) ---
try:
    from security import is_technician  # основной путь
except Exception:
    from flask_login import current_user
    def is_technician() -> bool:
        return (getattr(current_user, "role", "") or "").strip().lower() == "technician"



def _query_technicians():
    """Return list of (id, username) for users with 'technician' role."""
    return (
        db.session.query(User.id, User.username)
        .filter(func.lower(func.trim(User.role)) == "technician")
        .order_by(func.lower(func.trim(User.username)).asc())
        .all()
    )

# рядом с другими утилитами
def _is_superadmin_user(u) -> bool:
    role = (getattr(u, "role", "") or "").strip().lower()
    # поддерживаем все твои варианты флагов
    return (
        role in ("superadmin", "super admin")
        or bool(getattr(u, "is_superadmin", False))
        or bool(getattr(u, "is_super_admin", False))
    )


def _supplier_to_default_location(supplier_hint: str | None) -> str:
    """По имени/подсказке поставщика выбрать дефолтную локацию."""
    s = (supplier_hint or "").lower()
    for key, loc in SUPPLIER_LOC_DEFAULTS.items():
        if key in s:
            return loc
    return "MAIN"

def _apply_default_location(obj, default_loc: str):
    """
    Проставить default_loc туда, где location пустая.
    Работает и с DataFrame, и со списком dict.
    """
    try:
        import pandas as pd  # локальный импорт — не мешаем остальному
        if hasattr(obj, "copy") and hasattr(obj, "columns"):  # вероятно, DataFrame
            df = obj.copy()
            if "location" not in df.columns:
                df["location"] = ""
            df["location"] = df["location"].fillna("").replace("", default_loc)
            return df
    except Exception:
        pass

    # список словарей
    if isinstance(obj, list):
        for r in obj:
            if not (r.get("location") or "").strip():
                r["location"] = default_loc
        return obj

    return obj

def _resolve_default_loc(df, default_loc: str | None, saved_path: str, supplier_hint: str | None = None) -> str:
    """
    Надёжно выбираем дефолтную локацию:
    1) Явный default_loc (если передали)
    2) supplier_hint (из контента / формы)
    3) df['supplier'] (если распознался текстом)
    4) имя файла (contains 'reliable'/'marcone' и т.п.)
    5) MAIN
    """
    # 1) уже переданный default_loc
    if default_loc:
        return str(default_loc).strip().upper()

    # 2) supplier_hint
    loc = _supplier_to_default_location(supplier_hint)
    if loc and loc != "MAIN":
        return loc

    # 3) колонка supplier в DF
    try:
        if df is not None and "supplier" in df.columns:
            vals = " ".join(str(x) for x in df["supplier"].dropna().astype(str).tolist()).lower()
            loc3 = _supplier_to_default_location(vals)
            if loc3 and loc3 != "MAIN":
                return loc3
    except Exception:
        pass

    # 4) имя файла
    base = (saved_path or "").lower()
    loc4 = _supplier_to_default_location(base)
    if loc4 and loc4 != "MAIN":
        return loc4

    # 5) запасной вариант
    return "MAIN"


def _clear_dedup_keys_for_batch(batch_id: int, supplier: str | None = None, invoice: str | None = None) -> int:
    """
    Удаляет записи анти-дедупа, связанные с batch_id / supplier / invoice.
    Возвращает количество удалённых записей.
    Стратегии:
      1) ImportDedupKey (если есть).
      2) inventory.dedup_store (если есть).
      3) Рефлексия БД по 'dedup' + 'import' + 'processed' + 'parse' + 'lock' + 'invoice' таблицам.
         Матч по:
           - batch_id/receiving_batch_id/goods_receipt_id
           - supplier/supplier_name/vendor/vendor_name
           - invoice/invoice_number/inv_no/inv_number/invoice_no/order_no
           - key/hash/fingerprint
           - source_file/filename/file_path
           - JSON-поля (meta_json/meta/data) по batch_id/supplier/invoice
    """
    deleted = 0
    sup = (supplier or "").strip()
    inv = (invoice  or "").strip()

    # --- Стратегия 1: ImportDedupKey (если модель существует) ---
    try:
        from models import ImportDedupKey
        from sqlalchemy import or_, cast, String, func

        q = ImportDedupKey.query
        or_conds = []

        # JSON-поля: batch_id, supplier, invoice
        for mc in ("meta_json", "meta", "data"):
            if hasattr(ImportDedupKey, mc):
                col = getattr(ImportDedupKey, mc)
                or_conds.append(cast(col, String).ilike(f'%\"batch_id\": {batch_id}%'))
                or_conds.append(cast(col, String).ilike(f'%\"batch_id\":\"{batch_id}\"%'))
                if sup:
                    or_conds.append(cast(col, String).ilike(f'%\"supplier\":\"{sup}\"%'))
                    or_conds.append(cast(col, String).ilike(f'%\"supplier_name\":\"{sup}\"%'))
                if inv:
                    or_conds.append(cast(col, String).ilike(f'%\"invoice\":\"{inv}\"%'))
                    or_conds.append(cast(col, String).ilike(f'%\"invoice_number\":\"{inv}\"%'))
                    or_conds.append(cast(col, String).ilike(f'%\"invoice_no\":\"{inv}\"%'))
                    or_conds.append(cast(col, String).ilike(f'%\"order_no\":\"{inv}\"%'))

        # key/hash
        for name in ("key", "hash", "fingerprint"):
            if hasattr(ImportDedupKey, name):
                col = getattr(ImportDedupKey, name)
                if batch_id:
                    or_conds.append(col.ilike(f"%|B{batch_id}"))
                if sup and inv:
                    or_conds.append(func.upper(col) == (sup + inv).upper())

        # supplier/invoice прямыми колонками
        if sup and hasattr(ImportDedupKey, "supplier"):
            or_conds.append(func.upper(getattr(ImportDedupKey, "supplier")) == sup.upper())
        if sup and hasattr(ImportDedupKey, "supplier_name"):
            or_conds.append(func.upper(getattr(ImportDedupKey, "supplier_name")) == sup.upper())
        for inv_col in ("invoice_number", "invoice", "invoice_no", "order_no"):
            if inv and hasattr(ImportDedupKey, inv_col):
                or_conds.append(func.upper(getattr(ImportDedupKey, inv_col)) == inv.upper())

        # файловые колонки
        for file_col in ("source_file", "filename", "file_path"):
            if hasattr(ImportDedupKey, file_col):
                col = getattr(ImportDedupKey, file_col)
                # если в meta нет - всё равно почистим по наличию файла, см. откуда вызывает парсер
                # оставляем без условия — зависит от твоего кейса

        if or_conds:
            rows = q.filter(or_(*or_conds)).all()
            for row in rows:
                db.session.delete(row)
                deleted += 1
            if rows:
                db.session.commit()
    except ImportError:
        pass
    except Exception:
        db.session.rollback()

    # --- Стратегия 2: dedup_store API (если есть) ---
    try:
        try:
            from inventory.dedup_store import iter_keys, del_key  # type: ignore
        except Exception:
            from inventory.import_ledger import iter_keys, del_key  # fallback

        for k, meta in iter_keys():
            try:
                meta = meta or {}
                bid = str(meta.get("batch_id", ""))
                if bid == str(batch_id) or f"|B{batch_id}" in str(k):
                    if sup and str(meta.get("supplier", "")).strip():
                        if str(meta.get("supplier")).strip().upper() != sup.upper():
                            continue
                    if inv and str(meta.get("invoice", "")).strip():
                        if str(meta.get("invoice")).strip().upper() != inv.upper():
                            continue
                    if del_key(k):
                        deleted += 1
            except Exception:
                continue
    except Exception:
        pass

    # 2b) del_keys_by_batch
    try:
        try:
            from inventory.dedup_store import del_keys_by_batch  # type: ignore
        except Exception:
            from inventory.import_ledger import del_keys_by_batch  # fallback
        deleted += int(del_keys_by_batch(batch_id) or 0)
    except Exception:
        pass

    # --- Стратегия 3: расширенная рефлексия БД ---
    try:
        import sqlalchemy as sa
        engine = db.session.get_bind()
        insp   = sa.inspect(engine)

        TABLE_NAME_HINTS = ("dedup", "import", "processed", "parse", "lock", "invoice")
        POSSIBLE_BATCH   = ("batch_id", "goods_receipt_id", "receiving_batch_id")
        POSSIBLE_SUP     = ("supplier", "supplier_name", "vendor", "vendor_name")
        POSSIBLE_INV     = ("invoice", "invoice_number", "inv_no", "inv_number", "invoice_no", "order_no")
        POSSIBLE_KEY     = ("key", "hash", "fingerprint")
        POSSIBLE_FILE    = ("source_file", "filename", "file_path")
        META_CANDS       = ("meta_json", "meta", "data")

        for tname in insp.get_table_names():
            lname = tname.lower()
            if not any(h in lname for h in TABLE_NAME_HINTS):
                continue

            meta = sa.MetaData()
            try:
                table = sa.Table(tname, meta, autoload_with=engine)
            except Exception:
                continue

            cols = {c.name.lower(): c for c in table.columns}

            col_batch = next((cols[n] for n in POSSIBLE_BATCH if n in cols), None)
            col_sup   = next((cols[n] for n in POSSIBLE_SUP   if n in cols), None)
            col_inv   = next((cols[n] for n in POSSIBLE_INV   if n in cols), None)
            col_key   = next((cols[n] for n in POSSIBLE_KEY   if n in cols), None)
            col_file  = next((cols[n] for n in POSSIBLE_FILE  if n in cols), None)

            or_conds = []
            if col_batch is not None:
                or_conds.append(col_batch == int(batch_id))
            if sup and col_sup is not None:
                or_conds.append(sa.func.upper(col_sup) == sup.upper())
            if inv and col_inv is not None:
                or_conds.append(sa.func.upper(col_inv) == inv.upper())
            if col_key is not None:
                # два популярных паттерна ключей
                if batch_id:
                    or_conds.append(sa.cast(col_key, String).ilike(f"%|B{batch_id}"))
                if sup and inv:
                    or_conds.append(sa.func.upper(col_key) == (sup + inv).upper())
            # JSON-поля
            for mc in META_CANDS:
                if mc in cols:
                    jcol = cols[mc]
                    or_conds.append(sa.cast(jcol, String).ilike(f'%\"batch_id\": {batch_id}%'))
                    or_conds.append(sa.cast(jcol, String).ilike(f'%\"batch_id\":\"{batch_id}\"%'))
                    if sup:
                        or_conds.append(sa.cast(jcol, String).ilike(f'%\"supplier\":\"{sup}\"%'))
                        or_conds.append(sa.cast(jcol, String).ilike(f'%\"supplier_name\":\"{sup}\"%'))
                    if inv:
                        for invk in ("invoice", "invoice_number", "invoice_no", "order_no"):
                            or_conds.append(sa.cast(jcol, String).ilike(f'%\"{invk}\":\"{inv}\"%'))

            if not or_conds:
                continue

            stmt = sa.delete(table).where(sa.or_(*or_conds))
            try:
                res = engine.execute(stmt)
                deleted += int(getattr(res, "rowcount", 0) or 0)
            except Exception:
                continue

        if deleted:
            db.session.commit()
    except Exception:
        pass

    return int(deleted or 0)

@inventory_bp.post("/receiving/<int:batch_id>/delete", endpoint="receiving_delete")
@login_required
def receiving_delete(batch_id: int):
    from flask import current_app, flash, redirect, url_for
    from flask_login import current_user
    from extensions import db
    from sqlalchemy import func
    from models import ReceivingBatch
    from services.receiving import unpost_receiving_batch

    # --- доступ ---
    role = (getattr(current_user, "role", "") or "").lower()
    if not (
        role == "superadmin"
        or getattr(current_user, "is_superadmin", False)
        or getattr(current_user, "is_super_admin", False)
    ):
        flash("Only superadmin can delete receiving.", "danger")
        return redirect(url_for("inventory.receiving_list"))

    batch = db.session.get(ReceivingBatch, batch_id)
    if not batch:
        flash("Batch not found.", "warning")
        return redirect(url_for("inventory.receiving_list"))

    supplier = (getattr(batch, "supplier_name", "") or "").strip()
    invoice  = (getattr(batch, "invoice_number", "") or "").strip()
    status   = (getattr(batch, "status", "") or "").lower()

    # helper: вытащить строки batch-а (для логов, и чтобы потом не лазить в уже удалённый объект)
    def _get_lines_for_batch(b):
        try:
            return list(b.items or [])
        except Exception:
            pass
        try:
            from models import GoodsReceiptLine
            return list(GoodsReceiptLine.query.filter_by(goods_receipt_id=b.id).all())
        except Exception:
            return []

    lines = _get_lines_for_batch(batch)

    pns = {
        (getattr(it, "part_number", "") or "").strip().upper()
        for it in lines
        if (getattr(it, "part_number", "") or "").strip()
    }

    current_app.logger.info(
        "[DELETE RECEIVING] start: batch_id=%s, status=%s, supplier=%s, invoice=%s, pns=%s",
        batch_id, status, supplier, invoice, sorted(pns)
    )

    # 1. Если батч был posted → аккуратно откатываем СКЛАД через официальный сервис.
    #    ВАЖНО: это уже делает минус qty по каждой строке.
    #    После этого qty в инвентаре должен стать (старый - прихода этого батча).
    if status == "posted":
        try:
            unpost_receiving_batch(batch.id, getattr(current_user, "id", None))
            current_app.logger.info("[DELETE RECEIVING] unposted batch_id=%s", batch_id)
            # перечитаем batch после unpost (он теперь draft)
            batch = db.session.get(ReceivingBatch, batch_id)
        except Exception as e:
            current_app.logger.exception(
                "[DELETE RECEIVING] unpost failed for batch_id=%s",
                batch_id
            )
            flash(f"Unable to delete: auto-unpost failed: {e}", "danger")
            return redirect(url_for("inventory.receiving_detail", batch_id=batch_id))

    # 2. Если батч уже был draft (не posted), мы НИЧЕГО не трогаем в остатках.
    #    Почему? draft не должен быть в инвентаре. Если он как-то попал туда ранним багом —
    #    мы не будем делать второй минус. Это лучше, чем случайно обнулить сток.

    # 3. Теперь удаляем сам batch.
    try:
        batch = db.session.get(ReceivingBatch, batch_id)
        if batch:
            db.session.delete(batch)
        db.session.commit()
        current_app.logger.info("[DELETE RECEIVING] batch deleted batch_id=%s", batch_id)
    except Exception as e:
        db.session.rollback()
        current_app.logger.exception("[DELETE RECEIVING] delete failed for batch_id=%s", batch_id)
        flash(f"Failed to delete batch: {e}", "danger")
        return redirect(url_for("inventory.receiving_detail", batch_id=batch_id))

    # 4. Чистим dedup-ключи (чтоб можно было снова импортнуть тот же invoice/file).
    keys_removed = 0
    try:
        fn = globals().get("_clear_dedup_keys_for_batch")
        if fn:
            try:
                removed = fn(batch_id, supplier=supplier, invoice=invoice)
            except TypeError:
                removed = fn(batch_id)
            keys_removed = int(removed or 0)
            current_app.logger.info(
                "[DELETE RECEIVING] dedup removed=%s for batch_id=%s",
                keys_removed, batch_id
            )
        else:
            current_app.logger.info(
                "[DELETE RECEIVING] no _clear_dedup_keys_for_batch; skip dedup cleanup for batch_id=%s",
                batch_id
            )
    except Exception:
        current_app.logger.debug(
            "[DELETE RECEIVING] dedup cleanup failed for batch_id=%s",
            batch_id,
            exc_info=True
        )

    # 5. ВАЖНО: мы НЕ удаляем Part, и мы НЕ делаем никаких "вручную минусни qty".
    #    Всё, что должно списаться, уже списал unpost_receiving_batch().
    #    Значит:
    #      было 2 → постнули +1 → стало 3
    #      unpost → 3-1 = 2
    #      delete → просто убрали сам batch, остаток остался 2
    #    Именно то, что ты хочешь.

    msg = f"Batch #{batch_id} deleted."
    if keys_removed:
        msg += f" Dedup keys removed: {keys_removed}."

    flash(msg, "success")

    current_app.logger.info(
        "[DELETE RECEIVING] done batch_id=%s, keys_removed=%s",
        batch_id, keys_removed
    )

    return redirect(url_for("inventory.receiving_list"))

@inventory_bp.post("/receiving/<int:batch_id>/clear-keys", endpoint="receiving_clear_keys")
@login_required
def receiving_clear_keys(batch_id: int):
    from flask import flash, redirect, url_for
    from flask_login import current_user
    from extensions import db
    from models import ReceivingBatch

    role = (getattr(current_user, "role", "") or "").lower()
    if not (role == "superadmin" or getattr(current_user, "is_superadmin", False) or getattr(current_user, "is_super_admin", False)):
        return ("Forbidden", 403)

    batch = db.session.get(ReceivingBatch, batch_id)
    supplier = (getattr(batch, "supplier_name", "") or "").strip() if batch else ""
    invoice  = (getattr(batch, "invoice_number", "") or "").strip() if batch else ""

    try:
        removed = _clear_dedup_keys_for_batch(batch_id, supplier=supplier, invoice=invoice)
        if removed:
            flash(f"Dedup keys cleared for Batch #{batch_id} (removed: {removed}).", "success")
        else:
            flash(f"No dedup keys found for Batch #{batch_id}.", "info")
    except Exception as e:
        flash(f"Failed to clear keys: {e}", "danger")

    return redirect(url_for("inventory.receiving_list"))

# def _ensure_norm_columns(df, default_loc: str, saved_path: str):
#     import pandas as pd
#     df = _coerce_norm_df(df).copy()
#
#     # Базовый набор
#     need_cols = [
#         "part_number", "part_name",
#         "qty", "unit_cost", "location",
#         "row_key", "source_file",
#         "supplier", "order_no", "invoice_no"
#     ]
#     for c in need_cols:
#         if c not in df.columns:
#             df[c] = None
#
#     # qty → int, отрицательные в 0
#     df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0).astype(int)
#     df.loc[df["qty"] < 0, "qty"] = 0
#
#     # === ALIASES (ВАЖНО) =====================================================
#     # build_receive_movements ждёт 'quantity' → делаем алиас
#     df["quantity"] = df["qty"]
#
#     # На всякий случай поддержим разные имена цены
#     if "unit_cost" not in df.columns or df["unit_cost"].isna().all():
#         for alt in ["unitcost", "unit_price", "unitprice", "price", "cost", "last_cost"]:
#             if alt in df.columns and not df[alt].isna().all():
#                 df["unit_cost"] = df[alt]
#                 break
#     # ========================================================================
#
#     # unit_cost → float
#     df["unit_cost"] = pd.to_numeric(df["unit_cost"], errors="coerce")
#
#     # location: только пустые заполняем default_loc
#     empty_loc = df["location"].isna() | (df["location"].astype(str).str.strip() == "")
#     df.loc[empty_loc, "location"] = default_loc
#
#     # source_file: только пустые — путём загрузки
#     empty_sf = df["source_file"].isna() | (df["source_file"].astype(str).str.strip() == "")
#     df.loc[empty_sf, "source_file"] = saved_path
#
#     # row_key: генерим, если пусто
#     def _mk_key(row):
#         return f"{row.get('part_number','')}/{row.get('location','')}/{row.get('qty',0)}/{row.get('unit_cost','')}"
#     empty_rk = df["row_key"].isna() | (df["row_key"].astype(str).str.strip() == "")
#     df.loc[empty_rk, "row_key"] = df[empty_rk].apply(_mk_key, axis=1)
#
#     return df

# --- ЖЁСТКАЯ НОРМАЛИЗАЦИЯ ШАПОК ТАБЛИЦЫ ---
def _harden_preview_headers(df, default_loc: str, supplier_hint: str | None, source_file: str | None):
    """
    Гарантирует наличие и корректные типы колонок:
      part_number, part_name, quantity, unit_cost, location, supplier, source_file
    Понимает PDF-шапки: PART #, DESCRIPTION, QTY, UNIT COST, LOCATION.
    """
    import re
    import pandas as pd

    if df is None:
        return df

    # 1) точные заголовки из PDF
    hard_map = {
        "PART #": "part_number",
        "PART#": "part_number",
        "DESCRIPTION": "part_name",
        "QTY": "quantity",
        "UNIT COST": "unit_cost",
        "LOCATION": "location",
        "SUPPLIER": "supplier",
        "ORDER #": "order_no",
        "ORDER NO": "order_no",
        "PO": "order_no",
    }
    for col in list(df.columns):
        k = str(col).strip()
        if k.upper() in hard_map:
            df.rename(columns={col: hard_map[k.upper()]}, inplace=True)

    # 2) алиасы на всякий случай (вдруг normalize_table вернула иначе)
    alias = {
        "part_number": ["pn", "part", "number", "sku", "code", "item", "item #", "item#"],
        "part_name":   ["name", "descr", "description", "title"],
        "quantity":    ["qty", "count", "on_hand", "onhand", "q-ty"],
        "unit_cost":   ["cost", "price", "unitprice", "last_cost", "unit price", "unitprice"],
        "location":    ["loc", "bin", "shelf", "place"],
        "supplier":    ["vendor", "provider"],
        "order_no":    ["invoice", "invoice_no", "invoice number", "po"],
    }
    for canonical, cands in alias.items():
        if canonical not in df.columns:
            for c in cands:
                if c in df.columns:
                    df.rename(columns={c: canonical}, inplace=True)
                    break

    # 3) обязательные поля — если нет, создадим пустые
    for col in ("part_number", "part_name", "quantity", "unit_cost", "location", "supplier", "order_no", "source_file"):
        if col not in df.columns:
            df[col] = None

    # 4) если part_number пуст — попытка угадать по «похожести» на артикулы
    if df["part_number"].isna().all() or (df["part_number"].astype(str).str.strip() == "").all():
        candidate = None
        for c in df.columns:
            if c in ("part_number", "part_name", "quantity", "unit_cost", "location", "supplier", "order_no", "source_file"):
                continue
            s = df[c].astype(str).str.strip()
            mask = s.str.match(r"(?i)^(?=.*\d)[A-Z0-9\-\._/]{3,30}$")
            if mask.mean() >= 0.6:
                candidate = c
                break
        if candidate:
            df["part_number"] = df[candidate].astype(str).str.strip()

    # 5) типы
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0).astype(int)
    df["unit_cost"] = pd.to_numeric(df["unit_cost"], errors="coerce").fillna(0.0)

    for col in ("part_number", "part_name", "location", "supplier", "order_no", "source_file"):
        df[col] = df[col].astype(str).replace({"None": ""}).fillna("").str.strip()

    # 6) значения по умолчанию
    if not default_loc:
        default_loc = "MAIN"
    df.loc[df["location"].eq("") | df["location"].isna(), "location"] = default_loc
    df["location"] = df["location"].str.upper()

    if supplier_hint:
        df.loc[df["supplier"].eq("") | df["supplier"].isna(), "supplier"] = supplier_hint

    if source_file:
        # заполним пустые/пробельные source_file
        empty_sf = (df["source_file"].isna()) | (df["source_file"].astype(str).str.strip() == "")
        df.loc[empty_sf, "source_file"] = source_file

    # 7) отфильтруем пустые строки, но НЕ слишком агрессивно
    keep = (
        (~df["part_number"].astype(str).str.strip().eq("")) |
        (~df["part_name"].astype(str).str.strip().eq("")) |
        (df["quantity"] > 0) |
        (df["unit_cost"] > 0)
    )
    return df[keep].copy()



def _coerce_norm_df(obj) -> pd.DataFrame:
    """Гарантирует, что на выходе будет DataFrame. None/список/словарь → DF."""
    if isinstance(obj, pd.DataFrame):
        return obj
    if obj is None:
        return pd.DataFrame([])
    if isinstance(obj, list):
        return pd.DataFrame(obj)
    if isinstance(obj, dict):
        return pd.DataFrame([obj])
    return pd.DataFrame([])

# def _ensure_norm_columns(df, default_loc: str, saved_path: str):
#     """
#     Гарантирует обязательные колонки и аккуратно заполняет только ПУСТЫЕ значения.
#     НИЧЕГО не перезаписывает, если пользователь уже отредактировал ячейку.
#     """
#     import pandas as pd
#     import os
#
#     if df is None:
#         cols = [
#             "part_number","part_name","qty","quantity","unit_cost",
#             "location","row_key","source_file","supplier",
#             "order_no","invoice_no","date"
#         ]
#         return pd.DataFrame(columns=cols)
#
#     df = df.copy()
#
#     need_cols = [
#         "part_number","part_name","qty","quantity","unit_cost","location",
#         "row_key","source_file","supplier","order_no","invoice_no","date"
#     ]
#     for c in need_cols:
#         if c not in df.columns:
#             df[c] = None
#
#     # qty / quantity — синхронизация и мягкая типизация
#     qty = pd.to_numeric(df["qty"], errors="coerce")
#     quantity = pd.to_numeric(df["quantity"], errors="coerce")
#
#     df.loc[qty.isna() & quantity.notna(), "qty"] = quantity
#     df.loc[quantity.isna() & qty.notna(), "quantity"] = qty
#
#     df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0).astype(int)
#     df.loc[df["qty"] < 0, "qty"] = 0
#     df["quantity"] = df["qty"].astype(int)
#
#     # unit_cost — float (пустые оставляем NaN)
#     df["unit_cost"] = pd.to_numeric(df["unit_cost"], errors="coerce")
#
#     # source_file — только пустые
#     sf_empty = df["source_file"].isna() | (df["source_file"].astype(str).str.strip() == "")
#     df.loc[sf_empty, "source_file"] = saved_path
#
#     # location — только пустые; верхний регистр
#     loc_empty = df["location"].isna() | (df["location"].astype(str).str.strip() == "")
#     df.loc[loc_empty, "location"] = (default_loc or "MAIN")
#
#     df["location"] = (
#         df["location"]
#         .astype(str)
#         .str.strip()
#         .str.upper()
#     )
#
#     # строковые служебные
#     for col in ("part_number", "part_name", "supplier", "order_no", "invoice_no", "row_key", "source_file"):
#         df[col] = (
#             df[col]
#             .astype(str)
#             .replace({"None": ""})
#             .fillna("")
#             .str.strip()
#         )
#
#     # row_key — только где пусто (делаем устойчиво-уникальным на строку файла)
#     rk_empty = df["row_key"].isna() | (df["row_key"].astype(str).str.strip() == "")
#     if rk_empty.any():
#         df = df.reset_index(drop=False).rename(columns={"index": "__row_i"})
#         file_id = os.path.basename(str(saved_path or ""))
#
#         def _mk_key(row):
#             pn   = str(row.get("part_number", "")).strip().upper()
#             loc  = str(row.get("location", "")).strip().upper()
#             try:
#                 qty_local = int(pd.to_numeric(row.get("qty", 0), errors="coerce") or 0)
#             except Exception:
#                 qty_local = 0
#             cost = row.get("unit_cost", None)
#             if pd.isna(cost):
#                 cost = "NA"
#             else:
#                 try:
#                     cost = float(cost)
#                 except Exception:
#                     cost = "NA"
#             i = int(row.get("__row_i", 0) or 0)
#             return f"{file_id}|{i}|{pn}|{loc}|{qty_local}|{cost}"
#
#         df.loc[rk_empty, "row_key"] = df[rk_empty].apply(_mk_key, axis=1)
#         if "__row_i" in df.columns:
#             del df["__row_i"]
#
#     # подчистить полностью пустые строки
#     drop_mask = (
#         (df["part_number"].astype(str).str.strip() == "") &
#         (df["part_name"].astype(str).str.strip() == "") &
#         (df["qty"] == 0) &
#         (df["unit_cost"].fillna(0) == 0)
#     )
#     if drop_mask.any():
#         df = df[~drop_mask].copy()
#
#     return df

def parse_preview_rows_relaxed(form):
    buckets = defaultdict(dict)

    for key, val in form.items():
        m = _rows_re.match(key)
        if m:
            _unit_idx, row_idx, field = m.groups()
            buckets[int(row_idx)][field] = (val or "").strip()

    for key, val in form.items():
        m = _rows_flat_re.match(key)
        if m:
            row_idx, field = m.groups()
            buckets[int(row_idx)][field] = (val or "").strip()

    out = []
    for i in sorted(buckets.keys()):
        r = buckets[i]
        try:
            r["quantity"] = int((r.get("quantity") or "0").replace(",", ""))
        except Exception:
            r["quantity"] = 0
        try:
            r["unit_cost"] = float((r.get("unit_cost") or "0").replace("$", "").replace(",", ""))
        except Exception:
            r["unit_cost"] = 0.0

        if not any([(r.get("part_number") or "").strip(),
                    (r.get("part_name") or "").strip(),
                    r.get("quantity", 0),
                    r.get("unit_cost", 0.0)]):
            continue

        out.append({
            "part_number": (r.get("part_number") or "").strip(),
            "part_name":   (r.get("part_name") or "").strip(),
            "quantity":    r["quantity"],
            "unit_cost":   r["unit_cost"],
            "location":    (r.get("location") or "").strip(),
            "supplier":    (r.get("supplier") or "").strip(),
        })
    return out

def rows_to_norm_df(rows: list[dict], source_file: str):
    import pandas as pd
    out = []
    for i, r in enumerate(rows):
        pn = (r.get("part_number") or "").strip()
        if not pn:
            continue
        name = (r.get("part_name") or "").strip()
        loc  = (r.get("location") or "").strip()
        sup  = (r.get("supplier") or "").strip() or None

        # qty / unit_cost мягко
        q_raw = r.get("qty", r.get("quantity", 0))
        try:    qty = int(float(str(q_raw).strip() or 0))
        except: qty = 0

        u_raw = r.get("unit_cost", r.get("cost"))
        try:    unit_cost = float(str(u_raw).strip()) if u_raw not in (None, "") else None
        except: unit_cost = None

        out.append({
            "part_number": pn,
            "part_name": name,
            "qty": qty,
            "quantity": qty,
            "unit_cost": unit_cost,
            "location": loc,
            "supplier": sup,
            "source_file": source_file,
            "order_no": (r.get("order_no") or "").strip() or None,
            "invoice_no": (r.get("invoice_no") or "").strip() or None,
            "date": (r.get("date") or "").strip() or None,   # ← ДОБАВЛЕНО
            "row_key": f"{pn}|{loc or ''}|{i}",
        })
    return pd.DataFrame(out)

def parse_preview_rows_relaxed(form):
    """
    Собирает строки из request.form в формат:
    [{"part_number":..., "part_name":..., "quantity":..., "unit_cost":..., "location":...}, ...]
    Поддерживает оба имени: units[0][rows][i][field] И rows[i][field].
    Выкидывает пустые строки (без PN и Name и Qty и Cost).
    """
    buckets = defaultdict(dict)

    # 1) units[0][rows][i][field]
    for key, val in form.items():
        m = _rows_re.match(key)
        if m:
            _unit_idx, row_idx, field = m.groups()
            buckets[int(row_idx)][field] = (val or "").strip()

    # 2) rows[i][field] (fallback)
    for key, val in form.items():
        m = _rows_flat_re.match(key)
        if m:
            row_idx, field = m.groups()
            buckets[int(row_idx)][field] = (val or "").strip()

    # Сборка/очистка
    out = []
    for i in sorted(buckets.keys()):
        r = buckets[i]
        # нормализация типов
        try:
            r["quantity"] = int((r.get("quantity") or "0").replace(",", ""))
        except Exception:
            r["quantity"] = 0
        try:
            r["unit_cost"] = float((r.get("unit_cost") or "0").replace("$", "").replace(",", ""))
        except Exception:
            r["unit_cost"] = 0.0

        # пустые строки отбрасываем
        if not any([(r.get("part_number") or "").strip(),
                    (r.get("part_name") or "").strip(),
                    r.get("quantity", 0),
                    r.get("unit_cost", 0.0)]):
            continue

        out.append({
            "part_number": (r.get("part_number") or "").strip(),
            "part_name":   (r.get("part_name") or "").strip(),
            "quantity":    r["quantity"],
            "unit_cost":   r["unit_cost"],
            "location":    (r.get("location") or "").strip(),
            "supplier":    (r.get("supplier") or "").strip(),
        })
    return out

def fix_norm_records(records: list[dict], default_loc: str = "MAIN") -> list[dict]:
    """Правит записи уже ПОСЛЕ normalize_table: разлепляет PN/NAME, чистит мусор/итоги, дописывает LOCATION."""
    out = []
    for r in records:
        pn  = str(r.get("part_number", "") or "").strip()
        nm  = str(r.get("part_name", "") or "").strip()

        # 0) мусор и итоги (на всякий случай)
        if _MARKER_RX.match(pn) or _MARKER_RX.match(nm):
            continue
        if _TOTALS_RX.match(pn) or _TOTALS_RX.match(nm):
            continue

        # 1) PN выглядит как "PN rest..." → делим
        m = _PN_SPLIT_RX.match(pn)
        if m:
            tok, rest = m.group(1).strip(), m.group(2).strip()
            pn = tok
            if (not nm) or (nm.upper() == r.get("part_number","").upper()) or (nm.upper() == tok.upper()):
                nm = rest
            elif rest and rest not in nm:
                nm = f"{nm} {rest}".strip()

        # 2) PN пуст, а NAME = "PN rest..." → переносим
        elif not pn:
            m2 = _PN_SPLIT_RX.match(nm)
            if m2:
                pn, nm = m2.group(1).strip(), m2.group(2).strip()

        # 3) NAME дублирует PN → почистим
        if pn and nm and pn.upper() == nm.upper():
            nm = ""

        # 4) Локация по умолчанию
        loc = (r.get("location") or "").strip() or default_loc

        # 5) Записываем обратно
        r["part_number"] = pn
        r["part_name"]   = nm
        r["location"]    = loc

        # 6) Отсекаем полностью пустые
        if pn or nm or (r.get("quantity") or 0) or (r.get("unit_cost") or 0):
            out.append(r)
    return out

def fix_pn_and_description_in_df(df):
    if df is None or getattr(df, "empty", True):
        return df
    if "part_number" not in df.columns:
        df["part_number"] = ""
    if "part_name" not in df.columns:
        desc_col = None
        for k in df.columns:
            lk = k.lower().strip()
            if lk in ("description", "descr", "desc", "name", "part_name"):
                desc_col = k; break
        df["part_name"] = df[desc_col] if desc_col else ""

    pn = df["part_number"].astype(str); nm = df["part_name"].astype(str)
    new_pn, new_nm = pn.copy(), nm.copy()

    for i in df.index:
        cur_pn = (pn.at[i] or "").strip()
        cur_nm = (nm.at[i] or "").strip()

        m = _PN_FIRST_TOKEN_RX.match(cur_pn)
        if m:
            tok, rest = m.group(1).strip(), m.group(2).strip()
            new_pn.at[i] = tok
            if not cur_nm or cur_nm.upper() in (cur_pn.upper(), tok.upper()):
                new_nm.at[i] = rest
            elif rest and rest not in cur_nm:
                new_nm.at[i] = f"{cur_nm} {rest}"
            continue

        if not cur_pn:
            m2 = _PN_FIRST_TOKEN_RX.match(cur_nm)
            if m2:
                new_pn.at[i] = m2.group(1).strip()
                new_nm.at[i] = m2.group(2).strip()
                continue

        if cur_pn and cur_nm and cur_pn.upper() == cur_nm.upper():
            new_nm.at[i] = ""

    df["part_number"] = new_pn
    df["part_name"]   = new_nm
    return df

def _promote_split_pn_desc(df):
    """
    Если в исходной таблице PN склеен с описанием в одном поле
    (например: '60034 1/2FLRX1/2FIPANGG EZI'), создаём/заполняем
    явные колонки df['part_number'] и df['part_name'].
    Логика мягкая: применяем только если удаётся нормально сплитнуть
    заметную часть строк.
    """
    if df is None or getattr(df, "empty", True):
        return df

    # 1) найдём текстовые колонки-кандидаты
    text_cols = [c for c in df.columns if getattr(df[c], "dtype", None) == object]
    if not text_cols:
        return df

    # 2) если уже есть внятные part_number/part_name — уходим
    has_pn  = any(c.lower() == "part_number" for c in df.columns)
    has_name= any(c.lower() in ("part_name", "description", "descr", "desc", "name") for c in df.columns)
    if has_pn and has_name:
        return df  # не вмешиваемся

    # 3) пробуем найти колонку, где большинство строк выглядит как "TOKEN + пробел + текст"
    best_col = None
    best_score = 0
    for c in text_cols:
        series = df[c].astype(str).str.strip()
        m = series.str.match(_PN_TOKEN_RX, na=False)
        score = float(m.sum()) / max(1, len(series))
        if score > best_score:
            best_score, best_col = score, c

    # применяем только если >= 0.5 строк в колонке соответствует паттерну
    if not best_col or best_score < 0.5:
        return df

    # 4) делим выбранную колонку на PN + NAME
    pn_series  = []
    name_series= []
    for s in df[best_col].astype(str).tolist():
        s = s.strip()
        m = _PN_TOKEN_RX.match(s)
        if m:
            pn_series.append(m.group(1))
            name_series.append(m.group(2).strip())
        else:
            pn_series.append("")
            name_series.append(s)

    # 5) Создаём/заполняем явные колонки — так normalize_table возьмёт их без эвристик
    if "part_number" not in df.columns:
        df["part_number"] = pn_series
    else:
        # заполняем только пустые
        df["part_number"] = df["part_number"].astype(str).where(df["part_number"].astype(str).str.strip() != "", pn_series)

    # имя/описание
    # если есть 'part_name' — дополняем, иначе создаём
    if "part_name" in df.columns:
        cur = df["part_name"].astype(str)
        df["part_name"] = cur.where(cur.str.strip() != "", name_series)
    else:
        df["part_name"] = name_series

    return df

def drop_vendor_noise_rows(df):
    if df is None or getattr(df, "empty", True):
        return df
    text_cols = [c for c in df.columns if getattr(df[c], "dtype", None) == object]
    if not text_cols:
        return df
    has_marker = df[text_cols].apply(lambda s: s.astype(str).str.match(_MARKER_RX, na=False)).any(axis=1)
    return df[~has_marker].reset_index(drop=True)

def detect_supplier_hint(df, fname: str | None = None) -> str | None:
    fn = (fname or "").lower()
    if "reliable" in fn: return "Reliable"
    if "marcone"  in fn: return "Marcone"
    if df is not None and not df.empty:
        sample = " ".join(
            str(x).lower()
            for col in df.columns
            for x in df[col].head(50).astype(str).tolist()
        )
        if "reliable" in sample: return "Reliable"
        if "marcone"  in sample: return "Marcone"
    return None


def _coalesce_same_parts(rows: list[dict]) -> list[dict]:
    """
    Склеивает строки с одинаковым part_number (и одинаковой ценой),
    суммируя quantity. PN нормализуем в UPPER.
    """
    acc = {}
    for r in rows:
        pn = (r.get("part_number") or r.get("pn") or "").strip().upper()
        if not pn:
            continue
        price = float((r.get("unit_cost") or r.get("price") or 0) or 0)
        key = (pn, price)
        if key not in acc:
            acc[key] = {
                "part_number": pn,
                "part_name": (r.get("part_name") or r.get("description") or r.get("descr") or "").strip(),
                "quantity": 0,
                "unit_cost": price,
                "location": (r.get("location") or r.get("supplier") or "").strip(),
            }
        acc[key]["quantity"] += int((r.get("quantity") or r.get("qty") or 0) or 0)
    # отбрасываем пустые
    return [v for v in acc.values() if v["quantity"] > 0]



def _is_return_row(r) -> bool:
    """Возвратная строка — это строка с отрицательным количеством."""
    try:
        return (r.quantity or 0) < 0
    except Exception:
        return False

def _ensure_column(table: str, column: str, ddl_type: str):
    """Безопасно добавляет колонку, если её нет (SQLite).
       Если таблицы нет — просто логируем и выходим (без ошибок)."""
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


def _ensure_invoice_number_for_records(records, issued_to, issued_by, reference_job, issue_date, location):
    """
    Если у всех строк invoice_number == None — создаёт батч и закрепляет уникальный номер.
    Исторические инвойсы с уже заданным номером НЕ трогаем.
    """
    from extensions import db
    if not records:
        return

    if all(getattr(r, "invoice_number", None) is None for r in records):
        # Используем уже существующую у тебя функцию
        batch = _create_batch_for_records(
            records=records,
            issued_to=issued_to,
            issued_by=issued_by,
            reference_job=reference_job,
            issue_date=issue_date,
            location=location or None,
        )
        # номер закреплён в batch.invoice_number
        # flush/commit снаружи, в update_invoice


def _format_invoice_no(n: int | None) -> str:
    """UI/PDF формат: 6 цифр с ведущими нулями; '—' если None."""
    return f"{int(n):06d}" if n else "—"


def _tech_norm(s: str) -> str:
    return (s or '').strip().upper()

def _next_invoice_number() -> int:
    """
    Следующий invoice_number:
      max(IssuedBatch.invoice_number, IssuedPartRecord.invoice_number, INVOICE_START_AT-1) + 1
      (учитывает legacy-строки и гарантирует старт с 140)
    """
    # max по батчам (новая схема)
    max_batch = db.session.query(
        func.coalesce(func.max(IssuedBatch.invoice_number), 0)
    ).scalar()

    # max по строкам (legacy)
    max_line = db.session.query(
        func.coalesce(func.max(IssuedPartRecord.invoice_number), 0)
    ).scalar()

    try:
        mb = int(max_batch or 0)
    except Exception:
        mb = 0
    try:
        ml = int(max_line or 0)
    except Exception:
        ml = 0

    # базовый «семечко», чтобы первый новый = 140
    seed = INVOICE_START_AT - 1
    return max(mb, ml, seed) + 1


def _create_batch_for_records(
    records: list,
    issued_to: str,
    issued_by: str,
    reference_job: str | None = None,
    issue_date: datetime | None = None,
    location: str | None = None,
):
    """
    Создаёт IssuedBatch с уникальным invoice_number и привязывает все строки.
    Использует SAVEPOINT (begin_nested), чтобы коллизия unique не откатывала всю сессию.
    """
    if not records:
        raise ValueError("No records passed to _create_batch_for_records")

    issue_date = issue_date or datetime.utcnow()

    for _ in range(5):  # несколько попыток на случай гонки за номер
        inv_no = _next_invoice_number()

        try:
            with db.session.begin_nested():  # SAVEPOINT
                batch = IssuedBatch(
                    invoice_number=inv_no,
                    issued_to=issued_to,
                    issued_by=issued_by or "system",
                    reference_job=reference_job,
                    issue_date=issue_date,
                    location=(location or None),
                )
                db.session.add(batch)
                db.session.flush()  # резервируем уникальный номер (может кинуть IntegrityError)

                # привязываем строки к батчу и переносим «шапку» для консистентности
                for r in records:
                    r.batch_id = batch.id
                    r.invoice_number = inv_no
                    r.issued_to = issued_to
                    r.issued_by = issued_by or "system"
                    r.reference_job = reference_job
                    if location:
                        r.location = location

                db.session.flush()

            # успех
            return batch

        except IntegrityError:
            db.session.rollback()
            continue

    raise RuntimeError("Не удалось сгенерировать уникальный invoice_number после нескольких попыток")

def create_batch_for_records(records, issued_to, issued_by, reference_job=None, issue_date=None, location=None):
    """Создаёт IssuedBatch и привязывает к нему все переданные строки."""
    if not records:
        return None

    issue_date = issue_date or datetime.utcnow()

    # пробуем зарезервировать invoice_number с 2–3 попытками
    for _ in range(3):
        inv_no = _next_invoice_number()
        batch = IssuedBatch(
            invoice_number=inv_no,
            issued_to=issued_to,
            issued_by=issued_by,
            reference_job=reference_job,
            issue_date=issue_date,
            location=location
        )
        db.session.add(batch)

        try:
            db.session.flush()  # фиксируем уникальность invoice_number
        except IntegrityError:
            db.session.rollback()
            continue  # пробуем ещё раз с новым номером
        else:
            # Привязываем все строки
            for r in records:
                r.batch_id = batch.id
                r.invoice_number = inv_no  # оставляем дублирующий номер для отчётов
                r.issued_to = issued_to
                r.issued_by = issued_by
                r.reference_job = reference_job
                if location:
                    r.location = location
            return batch

    raise RuntimeError("Не удалось создать IssuedBatch: invoice_number конфликтует")




def _parse_dt_flex(s: str):
    """Безопасный парсер даты/времени.
       Поддерживает:
         - YYYY-MM-DD HH:MM:SS.%f
         - YYYY-MM-DD HH:MM:SS
         - YYYY-MM-DD
       Возвращает datetime или None.
    """
    if not s:
        return None
    s = str(s).strip()
    for fmt in ('%Y-%m-%d %H:%M:%S.%f',
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d'):
        try:
            dt = datetime.strptime(s, fmt)
            if fmt == '%Y-%m-%d':
                # Если только дата → прикрепляем полночь
                return datetime.combine(dt.date(), time.min)
            return dt
        except Exception:
            continue
    return None

def _parse_units_form(form):
    """
    Превращает плоский request.form во вложенную структуру:
    [
      {
        "brand": "...", "model": "...", "serial": "...",
        "rows": [
          {"part_number": "...", "part_name": "...", "quantity": 1, "unit_cost": 12.34, ...},
          ...
        ]
      },
      ...
    ]
    """
    units = {}

    # мета-поля юнитов (brand/model/serial)
    for k in form.keys():
        m = _units_re.match(k)
        if not m:
            continue
        ui = int(m.group(1))
        key = m.group(2)
        units.setdefault(ui, {"brand": "", "model": "", "serial": "", "rows": {}})
        units[ui][key] = (form.get(k) or "").strip()

    # строки
    for k in form.keys():
        m = _rows_re.match(k)
        if not m:
            continue
        ui = int(m.group(1))
        ri = int(m.group(2))
        key = m.group(3)

        units.setdefault(ui, {"brand": "", "model": "", "serial": "", "rows": {}})
        units[ui]["rows"].setdefault(ri, {
            "part_number": "", "part_name": "", "quantity": 0,
            "alt_numbers": "", "supplier": "", "backorder_flag": False,
            "line_status": "search_ordered", "unit_cost": ""
        })

        val = form.get(k)

        if key == "backorder_flag":
            units[ui]["rows"][ri][key] = True  # наличие ключа = checked
        else:
            units[ui]["rows"][ri][key] = (val or "").strip()

    # привести словарь → упорядоченный список
    result = []
    for ui in sorted(units.keys()):
        u = units[ui]
        # rows по порядку
        rows = [u["rows"][ri] for ri in sorted(u["rows"].keys())]
        u["rows"] = rows
        result.append(u)

    return result

@inventory_bp.post("/api/issued/confirm_toggle", endpoint="issued_confirm_toggle")
@login_required
def issued_confirm_toggle():
    """
    superadmin / admin:
        - могут ставить и убирать подтверждение.
    technician / tech:
        - могут только поставить подтверждение (one-way).
    остальные:
        - 403.

    После действия ВСЕГДА возвращаем пользователя обратно на страницу этого же Work Order,
    чтобы никого не выкидывало в общий список.
    """

    from flask import request, redirect, url_for, abort
    from datetime import datetime, timezone
    from flask_login import current_user
    from extensions import db
    from models import IssuedPartRecord

    role = (getattr(current_user, "role", "") or "").strip().lower()

    # Разрешённые роли
    if role not in ("superadmin", "admin", "technician", "tech"):
        abort(403)

    # Параметры формы
    try:
        rec_id = int(request.form.get("record_id") or 0)
    except Exception:
        rec_id = 0

    requested_state_is_checked = (request.form.get("state") == "1")

    wo_id = request.form.get("wo_id")

    # если нет wo_id, просто не рискуем и шлём 403 чтобы не гулять куда-то
    if not wo_id:
        abort(403)

    # Достаём запись
    rec = None
    if rec_id:
        rec = IssuedPartRecord.query.filter_by(id=rec_id).first()

    if not rec:
        # даже если не нашли запись - просто вернём обратно на ту же карточку WO
        return redirect(url_for("inventory.wo_detail", wo_id=wo_id))

    # текущее подтверждённое состояние в базе
    currently_confirmed = bool(rec.confirmed_by_tech)

    # Логика изменения в зависимости от роли
    if role in ("superadmin", "admin"):
        # полный контроль
        new_state = requested_state_is_checked

    elif role in ("technician", "tech"):
        # техник может только зафиксировать подтверждение, не снять
        if requested_state_is_checked:
            new_state = True
        else:
            new_state = currently_confirmed
    else:
        # защитный fallback - не должен сработать, но пусть будет
        abort(403)

    # Записываем новое состояние
    rec.confirmed_by_tech = new_state

    if new_state:
        # подтверждено
        rec.confirmed_by = (
            getattr(current_user, "username", "") or
            getattr(current_user, "email", "")
        )
        rec.confirmed_at = datetime.now(timezone.utc)
    else:
        # только admin/superadmin сюда попадают (они могут снимать)
        rec.confirmed_by = None
        rec.confirmed_at = None

    db.session.commit()

    # ВАЖНО: ВСЕГДА назад на конкретный Work Order. Никаких списков.
    return redirect(url_for("inventory.wo_detail", wo_id=wo_id))

@inventory_bp.get("/debug/db_objects")
def debug_db_objects():
    import os, sqlite3
    from flask import jsonify, current_app
    dbp = os.path.normpath(os.path.join(current_app.instance_path, "inventory.db"))
    data = {"db_path": dbp, "tables": [], "views": []}
    with sqlite3.connect(dbp) as con:
        data["tables"] = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()]
        data["views"] = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='view' ORDER BY name").fetchall()]
    return jsonify(data)


def _db_path() -> str:
    # Используем instance-папку Flask, там у тебя и лежит inventory.db
    p = os.path.join(current_app.instance_path, "inventory.db")
    return os.path.normpath(p)


# >>> ADD HELPERS (ONCE)
def _distinct_suppliers(wo_id: int):
    sql = """
    SELECT DISTINCT supplier
    FROM v_work_order_parts_dto
    WHERE wo_id = ? AND supplier <> ''
    ORDER BY supplier COLLATE NOCASE
    """
    dbp = _db_path()
    with sqlite3.connect(dbp) as con:
        con.row_factory = sqlite3.Row
        try:
            return [r[0] for r in con.execute(sql, (wo_id,)).fetchall()]
        except sqlite3.OperationalError as e:
            current_app.logger.error("distinct_suppliers error: %s", e)
            current_app.logger.error("DB_PATH = %s", dbp)
            exists = con.execute("""
                SELECT name FROM sqlite_master
                WHERE type IN ('view','table') AND name='v_work_order_parts_dto'
            """).fetchone()
            current_app.logger.error("v_work_order_parts_dto exists? %s", bool(exists))
            return []

def _issued_batches_stub(wo_id: int):
    return [{
        "issued_at": "2025-09-16 14:23",
        "technician": "Jane Doe",
        "canonical_ref": str(wo_id),
        "items": 3,
        "report_id": f"RPT-{wo_id}",
    }]


def _parse_q(q: str):
    if not q: return {"type":"text","value":""}
    m = _PREFIX_RE.match(q)
    return {"type": m.group(1).lower(), "value": m.group(2).strip()} if m else {"type":"text","value":q.strip()}

def get_on_hand(part_number: str) -> int:
    """Возвращает суммарный остаток по PN (всех локаций) из твоей модели Part."""
    try:
        total = db.session.query(db.func.sum(Part.quantity)).filter(
            Part.part_number == part_number.upper()
        ).scalar()
        return int(total or 0)
    except Exception:
        return 0

def compute_availability(work_order: "WorkOrder"):
    """
    Возвращает список словарей по каждой строке WO:
    [{
      wop_id, part_number, part_name, requested, on_hand, issue_now, status_hint
    }, ...]

    Логика:
      - Только при status == "ordered" обращаемся к складу:
          on_hand = get_on_hand(PN)
          issue_now = min(requested, on_hand)
          hint = "STOCK" | "WAIT {need} (stock {on})"
      - При других статусах (например, "search_ordered"):
          on_hand = 0, issue_now = 0, hint = "WAIT {requested} (not ordered)"
    """
    rows = []
    # считаем «можно ли проверять склад»
    check_stock = (getattr(work_order, "status", "") or "").lower() == "ordered"

    for wop in (work_order.parts or []):
        # нормализуем PN
        pn = (wop.part_number or "").strip().upper()
        # количество, безопасно
        req = int(wop.quantity or 0)
        if req < 0:
            req = 0

        if check_stock and pn:
            on = int(get_on_hand(pn) or 0)
            issue = min(req, on)
            if req <= 0:
                hint = "WAIT 0"
            elif on >= req:
                hint = "STOCK"
            else:
                need = req - on
                hint = f"WAIT {need} (stock {on})"
        else:
            # статус ещё не ordered → по твоей логике не сверяем склад
            on = 0
            issue = 0
            hint = f"WAIT {req} (not ordered)" if req > 0 else "WAIT 0"

        rows.append({
            "wop_id": wop.id,
            "part_number": pn,
            "part_name": (wop.part_name or "").strip(),
            "requested": req,
            "on_hand": on,
            "issue_now": issue,
            "status_hint": hint,
        })

    return rows

def compute_availability_unit(unit: "WorkUnit", wo_status: str):
    """
    Возвращает список строк по одному unit:
    [{unit_id, unit_label, part_number, part_name, requested, on_hand, issue_now, status_hint}, ...]
    """
    rows = []
    label = f"{(unit.brand or '').strip()} {(unit.model or '').strip()} / S/N {(unit.serial or '').strip()}".strip()
    label = label or f"UNIT #{unit.id}"

    for wup in (unit.parts or []):
        req = int(wup.quantity or 0)
        on  = get_on_hand((wup.part_number or "").upper())
        issue = 0
        # как договаривались: пока WO не в "ordered" — не выдаём, только WAIT ...
        if wo_status == "ordered":
            issue = max(0, min(req, on))
            hint = "STOCK" if on >= req else f"WAIT {req - on} (stock {on})"
        else:
            hint = f"WAIT {req} (not ordered)"

        rows.append({
            "unit_id": unit.id,
            "unit_label": label,
            "part_number": wup.part_number,
            "part_name": wup.part_name or "",
            "requested": req,
            "on_hand": on,
            "issue_now": issue,
            "status_hint": hint,
        })
    return rows

def compute_availability_multi(wo: "WorkOrder"):
    """
    Объединяет:
      - строки по юнитам (новая схема)
      - и твои старые строки work_order.parts (чтобы legacy не сломать)
    """
    all_rows = []

    # Новые unit-строки
    for unit in getattr(wo, "units", []) or []:
        all_rows.extend(compute_availability_unit(unit, wo.status))

    # Legacy-строки (если у заказа ещё есть старые parts)
    for wop in getattr(wo, "parts", []) or []:
        req = int(wop.quantity or 0)
        on  = get_on_hand((wop.part_number or "").upper())
        if wo.status == "ordered":
            issue = max(0, min(req, on))
            hint  = "STOCK" if on >= req else f"WAIT {req - on} (stock {on})"
        else:
            issue = 0
            hint  = f"WAIT {req} (not ordered)"
        all_rows.append({
            "unit_id": None,
            "unit_label": "LEGACY",
            "part_number": wop.part_number,
            "part_name": wop.part_name or "",
            "requested": req,
            "on_hand": on,
            "issue_now": issue,
            "status_hint": hint,
        })

    return all_rows

def _issue_records_bulk(issued_to: str, reference_job: str, items: list, billed_price_per_item: float | None = None):
    """
    items: list of dicts: { "part_id": int, "qty": int, ["unit_price": float] }
      - если unit_price передан в item — используем его как billed price (с учётом fee/markup),
        иначе возьмём текущий part.unit_cost.

    Возвращает (issue_date: datetime, created_count: int).
    """
    if not items:
        raise ValueError("No items to issue")

    issue_date = datetime.utcnow()
    created = 0

    for it in items:
        part_id = int(it["part_id"])
        qty     = max(0, int(it["qty"] or 0))
        if qty <= 0:
            continue

        part = Part.query.get(part_id)
        if not part:
            continue

        on_hand = int(part.quantity or 0)
        issue_now = min(qty, on_hand)
        if issue_now <= 0:
            continue

        # цена к фиксации — либо из item["unit_price"], либо текущая в складе
        price_to_fix = None
        if "unit_price" in it and it["unit_price"] is not None:
            price_to_fix = float(it["unit_price"])
        elif billed_price_per_item is not None:
            price_to_fix = float(billed_price_per_item)
        else:
            price_to_fix = float(part.unit_cost or 0.0)

        # уменьшаем склад
        part.quantity = on_hand - issue_now

        # создаём запись выдачи (твоя модель уже есть)
        rec = IssuedPartRecord(
            part_id=part.id,
            quantity=issue_now,
            issued_to=issued_to.strip(),
            reference_job=(reference_job or "").strip(),
            issued_by=current_user.username,
            issue_date=issue_date,
            unit_cost_at_issue=price_to_fix,  # фиксируем "billed" цену
        )
        db.session.add(rec)
        created += 1

    if created == 0:
        raise ValueError("Nothing available to issue")

    db.session.commit()
    return issue_date, created


# ==== RETURN HELPERS (без миграций) ====

def _is_return_record(record: IssuedPartRecord) -> bool:
    """Признак 'возвратной' строки — отрицательное количество или reference_job начинается с RETURN."""
    if record.quantity is not None and record.quantity < 0:
        return True
    ref = (record.reference_job or "").strip().upper()
    return ref.startswith("RETURN")

def _is_return_group(records: list[IssuedPartRecord]) -> bool:
    """Группа — возвратная, если есть хотя бы одна позиция с qty<0 или ref начинается с RETURN."""
    if not records:
        return False
    if any((r.quantity or 0) < 0 for r in records):
        return True
    ref = (records[0].reference_job or "").strip().upper()
    return ref.startswith("RETURN")

def _fetch_invoice_group(issued_to: str, reference_job: str|None, issued_by: str, issue_date):
    """Достаём ВСЕ записи накладной за конкретный день по ключу (issued_to, reference_job, issued_by, дата)."""
    from datetime import datetime
    start = datetime.combine(issue_date.date(), datetime.min.time())
    end   = datetime.combine(issue_date.date(), datetime.max.time())
    return IssuedPartRecord.query.filter(
        IssuedPartRecord.issued_to == issued_to,
        IssuedPartRecord.reference_job == reference_job,
        IssuedPartRecord.issued_by == issued_by,
        IssuedPartRecord.issue_date.between(start, end)
    ).all()
# ==== /RETURN HELPERS ====


# ---------- Helpers for editable preview ----------


def parse_preview_rows(form):
    """
    Parse rows[*][field] inputs from the preview form into a list of dicts.
    Accepts fields: part_number, part_name, quantity, unit_cost, supplier, location, order_no, date.
    Skips fully empty rows.
    """
    import re
    pattern = re.compile(r'^rows\[(\d+)\]\[(\w+)\]$')
    indexed = {}
    for key, val in form.items():
        m = pattern.match(key)
        if not m:
            continue
        idx = int(m.group(1))
        field = m.group(2)
        indexed.setdefault(idx, {})[field] = (val or "").strip()
    rows = []
    for i in sorted(indexed.keys()):
        r = indexed[i]
        # keep row if any meaningful cell is present
        if any((r.get(k) or "").strip() for k in ("part_number","part_name","quantity","unit_cost","supplier","location","order_no","date")):
            rows.append(r)
    return rows


def rows_to_dataframe(rows):
    """Turn list[dict] into a DataFrame with light type coercion."""
    import pandas as pd
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Ensure all expected columns exist
    for c in ["part_number","part_name","supplier","location","order_no","date","quantity","unit_cost"]:
        if c not in df.columns:
            df[c] = ""
    # Types
    df["quantity"]  = pd.to_numeric(df["quantity"], errors="coerce").fillna(0).astype(int)
    df["unit_cost"] = pd.to_numeric(df["unit_cost"], errors="coerce")
    for c in ["part_number","part_name","supplier","location","order_no","date"]:
        df[c] = df[c].astype(str).fillna("").str.strip()
    return df


# --- PDF items table fixer ----------------------------------------------------
def coerce_invoice_items(df_raw):
    """
    Find an invoice line-items header inside a noisy DataFrame (from pdfplumber)
    and rebuild a clean items table with canonical column names that your
    normalize_table() already understands.

    Canonical headers we try to produce:
      "QTY", "PART #", "DESCRIPTION", "UNIT COST", "TOTAL", plus optional
      "ORDER #", "DATE", "LOCATION".
    If nothing recognizable is found, returns df_raw unchanged.
    """
    import pandas as pd
    import numpy as np
    import re

    if df_raw is None or getattr(df_raw, "empty", True):
        return df_raw

    # Make a string-only view for scanning and cleanup (keeps original untouched)
    str_df = df_raw.copy()
    for j in range(str_df.shape[1]):
        s = str_df.iloc[:, j].astype(str).str.replace("\xa0", " ", regex=False)
        str_df.iloc[:, j] = s.fillna("").str.strip()

    # Canonical buckets (keys are the final column names we want to emit)
    BUCKETS = {
        "QTY":         ["QTY", "QUANTITY", "Q-ty", "QTY ORDERED", "ORDERED QTY"],
        "PART #":      ["PART #", "PART", "PART NO", "PART NUMBER", "ITEM", "ITEM #", "ITEM NO", "SKU", "MODEL", "MODEL #"],
        "DESCRIPTION": ["DESCRIPTION", "DESCR.", "ITEM DESCRIPTION", "DESC", "PRODUCT", "NAME"],
        "UNIT COST":   ["UNIT COST", "UNIT PRICE", "PRICE", "COST", "PRICE $", "PRICE USD"],
        "TOTAL":       ["TOTAL", "EXT", "EXTENDED", "LINE TOTAL", "AMOUNT"],

        # Optional extras — if present we keep them, else no problem
        "ORDER #":     ["ORDER #", "ORDER", "PO #", "SO #", "WORK ORDER", "WO"],
        "DATE":        ["DATE", "ORDER DATE", "INVOICE DATE"],
        "LOCATION":    ["LOCATION", "BIN", "SHELF", "PLACE", "LOC"],
    }

    def norm_key(s: str) -> str:
        # Normalize for fuzzy match: remove non-alphanumerics and uppercase
        return re.sub(r"[\W_]+", "", s.upper())

    # Pre-map all tokens to their bucket (final canonical name)
    token2bucket = {}
    for bucket_name, tokens in BUCKETS.items():
        for t in tokens:
            token2bucket[norm_key(t)] = bucket_name

    # Choose the "best" header row: the one covering the largest number of buckets
    best_idx, best_score = None, -1
    for i in range(str_df.shape[0]):
        seen = set()
        for val in str_df.iloc[i, :].tolist():
            k = norm_key(str(val))
            for token, bucket in token2bucket.items():
                if token and token in k:
                    seen.add(bucket)
        score = len(seen)
        if score > best_score:
            best_score, best_idx = score, i

    # If we cannot find a reasonable header (at least 2 buckets), bail out
    if best_idx is None or best_score < 2:
        return df_raw

    header_row = str_df.iloc[best_idx, :].tolist()

    # Map each header cell to a canonical bucket if possible
    def map_header_cell(s):
        ks = norm_key(str(s))
        for token, bucket in token2bucket.items():
            if token and token in ks:
                return bucket
        return ""  # unknown/unused

    mapped = [map_header_cell(v) for v in header_row]

    # Ensure uniqueness and fill blanks as col_<index>
    seen = {}
    final_cols = []
    for idx, m in enumerate(mapped):
        name = m if m else f"col_{idx}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        final_cols.append(name)

    # Body = rows under the header
    body = str_df.iloc[best_idx + 1 : ].copy()
    if body.empty:
        return df_raw

    # Rectangularize to header width
    if body.shape[1] < len(final_cols):
        for _ in range(len(final_cols) - body.shape[1]):
            body[f"pad_{_}"] = ""
    body = body.iloc[:, :len(final_cols)]
    body.columns = final_cols

    # Light type coercion for filters
    def to_int_safe(x):
        try:
            sx = str(x).replace(",", "").strip()
            if sx == "": return np.nan
            return int(float(sx))
        except:
            return np.nan

    def to_float_safe(x):
        try:
            sx = str(x).replace("$", "").replace(",", "").strip()
            if sx == "": return np.nan
            return float(sx)
        except:
            return np.nan

    if "QTY" in body.columns:
        body["QTY_num"] = body["QTY"].apply(to_int_safe)
    if "UNIT COST" in body.columns:
        body["UNIT_COST_num"] = body["UNIT COST"].apply(to_float_safe)

    # Keep rows that look like real items:
    keep_mask = False
    if "QTY_num" in body.columns:
        keep_mask = (body["QTY_num"].notna()) | (body["QTY"].astype(str).str.len() > 0)
    if "UNIT_COST_num" in body.columns:
        keep_mask = keep_mask | (body["UNIT_COST_num"].notna())
    if "PART #" in body.columns:
        keep_mask = keep_mask | (body["PART #"].astype(str).str.strip() != "")
    if "DESCRIPTION" in body.columns:
        keep_mask = keep_mask | (body["DESCRIPTION"].astype(str).str.strip() != "")

    if hasattr(keep_mask, "any"):
        body = body[keep_mask]

    # Drop fully empty rows
    body = body.replace("", np.nan).dropna(how="all").fillna("")

    return body if not body.empty else df_raw



def dataframe_from_pdf(path, try_ocr: bool = False):
    """
    1) Try extracting tables from a text-based PDF (pdfplumber).
    2) If no tables and try_ocr=True: OCR with poppler+Tesseract and then:
       - try to parse Marcone-like rows into PART # / DESCR. / QTY / UNIT COST / SUPPLIER
       - fallback to a generic table (at least 'text' column) so preview is never empty.
    """
    import os, re
    import pdfplumber
    import pandas as pd
    import numpy as np
    from flask import current_app
    from config import Config

    def log(msg: str):
        try:
            current_app.logger.debug(msg)
        except Exception:
            pass

    def unique_headers(headers):
        seen = {}
        out = []
        for i, h in enumerate(headers):
            h = (h or "").strip() or f"col_{i}"
            if h in seen:
                seen[h] += 1
                h = f"{h}_{seen[h]}"
            else:
                seen[h] = 0
            out.append(h)
        return out

    # ---------- A) TEXT-PDF ----------
    frames = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables(table_settings={
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines",
                    "intersection_tolerance": 5,
                }) or []
                if not tables:
                    tables = page.extract_tables(table_settings={
                        "vertical_strategy": "text",
                        "horizontal_strategy": "text",
                        "snap_tolerance": 3,
                        "intersection_tolerance": 3,
                        "join_tolerance": 3,
                    }) or []

                for t in tables:
                    rows = [r for r in t if any((c or "").strip() for c in r)]
                    if len(rows) < 2:
                        continue
                    w = max(len(r) for r in rows)
                    rows = [list(r) + [""] * (w - len(r)) for r in rows]

                    header = unique_headers([(h or "").strip() for h in rows[0]])
                    df = pd.DataFrame(rows[1:], columns=header)

                    for j in range(df.shape[1]):
                        s = df.iloc[:, j].astype(str)
                        s = s.str.replace("\xa0", " ", regex=False).str.strip()
                        s = s.mask(s.str.lower().isin(["nan", "none", "null"]), "")
                        df.iloc[:, j] = s

                    df = df.replace("", np.nan).dropna(how="all").fillna("")
                    if not df.empty:
                        frames.append(df)
    except Exception as e:
        log(f"[PDF] text-parse failed: {e}")

    if frames:
        out = pd.concat(frames, ignore_index=True)
        log(f"[PDF] text tables found, shape={out.shape}")
        return out

    # ---------- B) OCR ----------
    if not try_ocr:
        log("[OCR] try_ocr=False → skipping OCR")
        return pd.DataFrame()

    from pdf2image import convert_from_path
    import pytesseract

    poppler_bin = current_app.config.get("POPPLER_BIN") or getattr(
        Config, "POPPLER_BIN", r"C:\Program Files\poppler\bin"
    )
    tess_exe = current_app.config.get("TESSERACT_EXE") or getattr(
        Config, "TESSERACT_EXE", r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )
    pytesseract.pytesseract.tesseract_cmd = tess_exe

    try:
        images = convert_from_path(path, dpi=400, poppler_path=poppler_bin)
        log(f"[OCR] pages={len(images)} poppler_bin={poppler_bin}")
    except Exception as e:
        log(f"[OCR] convert_from_path failed: {e}")
        # вернём спец-таблицу, чтобы в превью видно было причину
        return pd.DataFrame({"error": [f"OCR init failed: {e}"]})

    tesseract_cfg = r"--oem 3 --psm 6"
    lines: list[str] = []
    for idx, img in enumerate(images, start=1):
        try:
            text = pytesseract.image_to_string(img, lang="eng", config=tesseract_cfg)
        except Exception as e:
            log(f"[OCR] page {idx} failed: {e}")
            continue
        for ln in text.splitlines():
            s = (ln or "").strip()
            if s:
                lines.append(s)

    # Dump raw OCR text (удобно проверять глазами)
    dump_path = os.path.splitext(path)[0] + ".ocr.txt"
    try:
        with open(dump_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        log(f"[OCR] lines={len(lines)} dump={dump_path}")
    except Exception:
        pass

    # ---- C) Try parsing Marcone-style rows
    base = os.path.basename(path).lower()
    supplier_guess = "Marcone" if "marcone" in base else ("ReliableParts" if "reliable" in base else "")

    money = r"\$[\d,]+\.\d{2}"
    pat_marcone = re.compile(
        rf"""
        ^\s*
        (?P<qty>\d+)\s+                      # qty
        (?P<pn>[A-Za-z0-9\-\/\.]+)\s+        # part number
        (?P<descr>.+?)\s+                    # description
        (?P<unit>{money})\s+                 # unit price
        (?:{money}\s+)?                      # optional MSRP/total
        (?P<price>{money})\s+                # total (ignored)
        (?P<bo>\d+)\s+(?P<ship>\d+)          # backorder / shipped (ignored)
        \s*$""",
        re.VERBOSE,
    )

    rows = []
    for ln in lines:
        m = pat_marcone.match(ln)
        if not m:
            continue
        rows.append({
            "PART #": m.group("pn").strip(),
            "DESCR.": m.group("descr").strip(),
            "QTY": int(m.group("qty")),
            "UNIT COST": float(m.group("unit").replace("$", "").replace(",", "")),
            "SUPPLIER": supplier_guess or "Marcone",
        })

    if rows:
        df = pd.DataFrame(rows)
        for c in ["PART #", "DESCR.", "SUPPLIER"]:
            df[c] = df[c].astype(str).str.replace("\xa0", " ", regex=False).str.strip()
        log(f"[OCR-MARCONE] parsed rows={len(df)}")
        return df

    # ---- D) Generic heuristic: "<qty> <pn> <descr...> <price$>" ИЛИ "<pn> <descr...> <qty> <price$>"
    pat1 = re.compile(rf"^\s*(?P<qty>\d+)\s+(?P<pn>[A-Za-z0-9\-\/\.]+)\s+(?P<descr>.+?)\s+(?P<unit>{money})\s*$")
    pat2 = re.compile(rf"^\s*(?P<pn>[A-Za-z0-9\-\/\.]+)\s+(?P<descr>.+?)\s+(?P<qty>\d+)\s+(?P<unit>{money})\s*$")

    rows = []
    for ln in lines:
        m = pat1.match(ln) or pat2.match(ln)
        if not m:
            continue
        qty = int(m.group("qty"))
        pn = m.group("pn").strip()
        descr = m.group("descr").strip()
        unit = float(m.group("unit").replace("$", "").replace(",", ""))
        rows.append({
            "PART #": pn,
            "DESCR.": descr,
            "QTY": qty,
            "UNIT COST": unit,
            "SUPPLIER": supplier_guess or "OCR",
        })

    if rows:
        df = pd.DataFrame(rows)
        for c in ["PART #", "DESCR.", "SUPPLIER"]:
            df[c] = df[c].astype(str).str.replace("\xa0", " ", regex=False).str.strip()
        log(f"[OCR-GENERIC] parsed rows={len(df)}")
        return df

    # ---- E) Last resort: show raw text so preview is not empty
    if lines:
        log("[OCR] fallback to raw text table")
        return pd.DataFrame({"text": lines})

    log("[OCR] no lines recognized")
    return pd.DataFrame()

def _settings_path():
    return os.path.join(current_app.instance_path, "app_settings.json")

def get_setting(key, default=None):
    try:
        with open(_settings_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {}
    return data.get(key, default)

def set_setting(key, value):
    p = _settings_path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {}
    data[key] = value
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# --- Confirm selected lines / whole invoice ---
@inventory_bp.post("/work_orders/<int:wo_id>/confirm")
@login_required
def wo_confirm_lines(wo_id: int):
    from flask import request, flash, redirect, url_for
    from datetime import datetime
    from sqlalchemy import func, or_
    from flask_login import current_user
    from extensions import db
    from models import WorkOrder, IssuedPartRecord, IssuedBatch

    wo = db.session.get(WorkOrder, wo_id)
    if not wo:
        flash(f"Work Order #{wo_id} not found.", "danger")
        return redirect(url_for("inventory.wo_list"))

    role = (current_user.role or "").lower()
    is_admin_like = role in ("admin", "superadmin")

    # проверка «свой ли WO» для техника
    if role == "technician":
        me_id = getattr(current_user, "id", None)
        me_name = (current_user.username or "").strip().lower()
        wo_tech_id = getattr(wo, "technician_id", None)
        wo_name = (wo.technician_username or wo.technician_name or "").strip().lower()
        is_my_wo = (
            (wo_tech_id and me_id and wo_tech_id == me_id)
            or (me_name and wo_name and me_name == wo_name)
        )
        if not is_my_wo:
            flash("Access denied", "danger")
            return redirect(url_for("inventory.wo_detail", wo_id=wo_id))

    # алиасы имени техника — чтобы совпало с issued_to
    me_aliases = {
        (current_user.username or "").strip().lower(),
        (wo.technician_username or "").strip().lower(),
        (wo.technician_name or "").strip().lower(),
    }
    me_aliases = {a for a in me_aliases if a}

    tech_display = (current_user.username or "").strip()
    if not tech_display:
        flash("Your account has no username; cannot confirm.", "danger")
        return redirect(url_for("inventory.wo_detail", wo_id=wo_id))

    canonical_job = (wo.canonical_job or "").strip()

    raw_ids = request.form.getlist("record_ids[]") or request.form.getlist("record_ids")
    sel_ids = [int(x) for x in raw_ids if str(x).isdigit()]
    inv_s = (request.form.get("invoice_number") or "").strip()
    inv_no = int(inv_s) if inv_s.isdigit() else None

    q = db.session.query(IssuedPartRecord).outerjoin(
        IssuedBatch, IssuedBatch.id == IssuedPartRecord.batch_id
    )

    if canonical_job:
        q = q.filter(
            or_(
                func.trim(IssuedPartRecord.reference_job) == canonical_job,
                func.trim(IssuedPartRecord.reference_job).like(f"%{canonical_job}%"),
                func.trim(IssuedBatch.reference_job) == canonical_job,
            )
        )

    if sel_ids:
        q = q.filter(IssuedPartRecord.id.in_(sel_ids))
    elif inv_no is not None:
        q = q.filter(IssuedPartRecord.invoice_number == inv_no)
    else:
        flash("Nothing to confirm (no IDs or invoice number).", "warning")
        return redirect(url_for("inventory.wo_detail", wo_id=wo_id))

    if not is_admin_like and me_aliases:
        # техник подтверждает только свои строки (без регистра)
        q = q.filter(func.lower(func.trim(IssuedPartRecord.issued_to)).in_(list(me_aliases)))

    rows = q.all()
    if not rows:
        flash("No matching lines found to confirm.", "warning")
        return redirect(url_for("inventory.wo_detail", wo_id=wo_id))

    now = datetime.utcnow()
    updated = 0
    for r in rows:
        if not getattr(r, "confirmed_by_tech", False):
            r.confirmed_by_tech = True
            r.confirmed_at = now
            r.confirmed_by = tech_display
            updated += 1

    try:
        db.session.commit()
        msg = f"Confirmed {updated} line(s)"
        if inv_no is not None:
            msg += f" in invoice #{inv_no:06d}"
        flash(msg + ("" if updated else " (no changes)."), "success" if updated else "info")
    except Exception as e:
        db.session.rollback()
        flash(f"Database error: {e}", "danger")

    return redirect(url_for("inventory.wo_detail", wo_id=wo_id))

@inventory_bp.get("/work_orders/new", endpoint="wo_new")
@login_required
def wo_new():
    role = (getattr(current_user, "role", "") or "").strip().lower()
    if role not in ("admin", "superadmin"):
        flash("Access denied", "danger")
        return redirect(url_for("inventory.wo_list"))

    # minimal empty WO-like object for the form
    class _WO:
        id = None
        technician_id = None
        technician_name = ""
        brand = ""
        model = ""
        serial = ""
        job_numbers = ""
        job_type = "BASE"
        delivery_fee = 0.0
        markup_percent = 0.0
        status = "search_ordered"
    wo = _WO()

    technicians = _query_technicians()
    recent_suppliers = session.get("recent_suppliers", []) or []

    # a single empty unit+row for the form
    units = [{
        "brand":  "",
        "model":  "",
        "serial": "",
        "rows": [{
            "id": None,
            "part_number": "", "part_name": "", "quantity": 1,
            "alt_numbers": "",
            "warehouse": "", "supplier": "",
            "backorder_flag": False, "line_status": "search_ordered",
            "unit_cost": 0.0,
        }],
    }]

    return render_template(
        "wo_form_units.html",
        wo=wo,
        units=units,
        recent_suppliers=recent_suppliers,
        readonly=False,
        technicians=technicians,
        selected_tech_id=None,
        selected_tech_username=None,
    )


@inventory_bp.get("/api/technicians")
@login_required
def api_technicians():
    from models import WorkOrder
    q = (request.args.get("q") or "").strip().upper()

    qry = (db.session.query(WorkOrder.technician_name)
           .filter(WorkOrder.technician_name.isnot(None),
                   WorkOrder.technician_name != ""))

    if q:
        qry = qry.filter(db.func.upper(WorkOrder.technician_name).like(q + "%"))

    names = sorted({(r[0] or "").strip().upper() for r in qry.limit(25).all() if r[0]})
    return jsonify({"items": names}), 200


@inventory_bp.get("/debug/db")
@login_required
def debug_db():
    import sqlite3, json, os
    from flask import jsonify, current_app
    dbp = os.path.join(current_app.instance_path, "inventory.db")
    dbp = os.path.normpath(dbp)
    views = []
    try:
        with sqlite3.connect(dbp) as con:
            views = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type IN ('view','table')").fetchall()]
    except Exception as e:
        return jsonify({"db_path": dbp, "error": str(e)})
    return jsonify({"db_path": dbp, "objects": views})



@inventory_bp.get("/search")
@login_required
def search():
    q = (request.args.get("q") or "").strip()
    status = (request.args.get("status") or "").strip()
    supplier = (request.args.get("supplier") or "").strip()
    scope = (request.args.get("scope") or "global").strip()
    wo_id = request.args.get("wo_id", type=int)
    limit = min(max(request.args.get("limit", type=int) or 50, 1), 200)
    offset = max(0, request.args.get("offset", type=int) or 0)

    if scope == "order" and not wo_id:
        abort(400, "wo_id required when scope=order")

    crit = _parse_q(q)
    where, params = [], {"limit":limit, "offset":offset}

    if scope == "order": where.append("wo_id=:wo_id"); params["wo_id"]=wo_id
    if status:           where.append("status=:status"); params["status"]=status
    if supplier:         where.append("supplier=:supplier COLLATE NOCASE"); params["supplier"]=supplier

    if crit["type"]=="text" and crit["value"]:
        params["v"]=f"%{crit['value']}%"
        where.append("(pn LIKE :v OR part_name LIKE :v OR brand LIKE :v OR model LIKE :v OR serial LIKE :v)")
    elif crit["type"]=="pn":
        params["pn"]=crit["value"].replace("-","").upper()+"%"
        where.append("UPPER(REPLACE(pn,'-','')) LIKE :pn")
    elif crit["type"] in {"brand","model","serial"}:
        params["v"]=f"%{crit['value']}%"; where.append(f"{crit['type']} LIKE :v")
    elif crit["type"] in {"wo","job"}:
        try: params["qwo"]=int(crit["value"]); where.append("wo_id=:qwo")
        except ValueError: pass

    where_sql=" WHERE "+ " AND ".join(where) if where else ""
    sql=f"""
      SELECT id, wo_id, pn, part_name, brand, model, serial,
             qty_needed, qty_issued, to_issue, unit_cost,
             on_hand, ordered_qty, eta, location, supplier, status
      FROM v_work_order_parts_dto
      {where_sql}
      ORDER BY wo_id DESC, brand, model, serial, pn
      LIMIT :limit OFFSET :offset
    """

    with sqlite3.connect(_db_path()) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(sql, params).fetchall()

    return jsonify({"items":[dict(r) for r in rows],
                    "limit":limit,"offset":offset,"scope":scope,"wo_id":wo_id})

@inventory_bp.get("/inventory/search", endpoint="search_alias")
@login_required
def search_alias():
    # просто проксируем на новый обработчик, сохраняя query params
    return search()

# --- toggle "Ordered" for a single WorkOrderPart ---
@inventory_bp.post("/work_orders/<int:wo_id>/parts/<int:wop_id>/toggle_ordered", endpoint="wo_toggle_ordered")
@login_required
def wo_toggle_ordered(wo_id: int, wop_id: int):
    from flask import request, flash, redirect, url_for
    from datetime import date
    from extensions import db
    from models import WorkOrderPart, WorkOrder
    from flask_login import current_user

    wop = db.session.get(WorkOrderPart, wop_id)
    if not wop or getattr(wop, "work_order_id", None) != wo_id:
        flash("Part row not found for this Work Order.", "danger")
        return redirect(url_for("inventory.wo_detail", wo_id=wo_id))

    wo = db.session.get(WorkOrder, wo_id)
    me   = (getattr(current_user, "username", "") or "").strip().lower()
    role = (getattr(current_user, "role", "") or "").strip().lower()
    tech = (getattr(wo, "technician_name", "") or "").strip().lower()

    if not (role in ("admin", "superadmin") or (me and me == tech)):
        flash("You are not allowed to change ordered status for this WO.", "danger")
        return redirect(url_for("inventory.wo_detail", wo_id=wo_id))

    new_state = (request.form.get("state") or "0").strip().lower() in ("1","true","on","yes")

    try:
        # sync флаги/статусы
        if hasattr(wop, "ordered_flag"):
            wop.ordered_flag = bool(new_state)
        if hasattr(wop, "status"):
            wop.status = "ordered" if new_state else "search_ordered"
        if hasattr(wop, "line_status"):
            wop.line_status = "ordered" if new_state else "search_ordered"

        # дата:
        if hasattr(wop, "ordered_date"):
            if new_state:
                # если даты нет — поставим сегодня; если есть — не трогаем
                if not wop.ordered_date:
                    wop.ordered_date = date.today()
            else:
                # снятие чекбокса — чистим дату
                wop.ordered_date = None

        db.session.commit()
        flash(("Ordered set" if new_state else "Ordered cleared") + " for the part.", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"Failed to update: {e}", "danger")

    return redirect(url_for("inventory.wo_detail", wo_id=wo_id))

# --- Work Order details ---
@inventory_bp.get("/work_orders/<int:wo_id>", endpoint="wo_detail")
@login_required
def wo_detail(wo_id):
    from flask import current_app, render_template, flash, redirect, url_for
    from sqlalchemy import func, or_, case
    from sqlalchemy.orm import selectinload, joinedload
    from flask_login import current_user
    from extensions import db
    from models import WorkOrder, WorkUnit, WorkOrderPart, Part

    # 1) Load Work Order with parts/units
    wo = (
        db.session.query(WorkOrder)
        .options(
            selectinload(WorkOrder.parts),
            selectinload(WorkOrder.units).selectinload(WorkUnit.parts),
        )
        .get(wo_id)
    )
    if not wo:
        flash(f"Work Order #{wo_id} not found.", "danger")
        return redirect(url_for("inventory.wo_list"))

    # 2) Permissions
    role = (getattr(current_user, "role", "") or "").strip().lower()
    me_id = getattr(current_user, "id", None)
    me_name = (getattr(current_user, "username", "") or "").strip().lower()
    wo_tech_id = getattr(wo, "technician_id", None)
    wo_tech_name = (wo.technician_username or wo.technician_name or "").strip().lower()

    is_admin_like = role in ("admin", "superadmin")
    is_technician = role == "technician"
    is_my_wo = (
        (wo_tech_id and me_id and wo_tech_id == me_id)
        or (me_name and wo_tech_name and me_name == wo_tech_name)
    )

    # Technician can only see their own WO
    if is_technician and not is_my_wo:
        flash("You don't have access to this Work Order.", "danger")
        return redirect(url_for("inventory.wo_list"))

    # Only superadmin can confirm batches
    can_confirm_any = (role == "superadmin")
    # Docs (Report/Print) visible to admin/superadmin/owner
    can_view_docs = (is_admin_like or is_my_wo)

    # 3) Suppliers list for this WO (unique, cleaned)
    suppliers = [
        r[0]
        for r in (
            db.session.query(func.trim(WorkOrderPart.supplier))
            .filter(
                WorkOrderPart.work_order_id == wo.id,
                WorkOrderPart.supplier.isnot(None),
                func.trim(WorkOrderPart.supplier) != "",
            )
            .distinct()
            .order_by(func.trim(WorkOrderPart.supplier).asc())
            .all()
        )
    ]

    # 4) Issued / Batches information
    from models import IssuedPartRecord, IssuedBatch  # noqa: E402
    from collections import defaultdict

    canon = (wo.canonical_job or "").strip()

    base_q = (
        db.session.query(IssuedPartRecord)
        .options(joinedload(IssuedPartRecord.part), joinedload(IssuedPartRecord.batch))
        .outerjoin(IssuedBatch, IssuedBatch.id == IssuedPartRecord.batch_id)
    )
    if canon:
        base_q = base_q.filter(
            or_(
                func.trim(IssuedPartRecord.reference_job) == canon,
                func.trim(IssuedPartRecord.reference_job).like(f"%{canon}%"),
                func.trim(IssuedBatch.reference_job) == canon,
            )
        )

    issued_items = base_q.order_by(
        IssuedPartRecord.issue_date.asc(), IssuedPartRecord.id.asc()
    ).all()

    # aggregates for Issued Items summary
    money_issued = func.coalesce(
        func.sum(
            case(
                (
                    IssuedPartRecord.quantity > 0,
                    IssuedPartRecord.quantity * IssuedPartRecord.unit_cost_at_issue,
                ),
                else_=0,
            )
        ),
        0.0,
    )
    money_returned = func.coalesce(
        func.sum(
            case(
                (
                    IssuedPartRecord.quantity < 0,
                    -IssuedPartRecord.quantity * IssuedPartRecord.unit_cost_at_issue,
                ),
                else_=0,
            )
        ),
        0.0,
    )
    qty_issued = func.coalesce(
        func.sum(
            case(
                (IssuedPartRecord.quantity > 0, IssuedPartRecord.quantity),
                else_=0,
            )
        ),
        0,
    )
    qty_returned = func.coalesce(
        func.sum(
            case(
                (IssuedPartRecord.quantity < 0, -IssuedPartRecord.quantity),
                else_=0,
            )
        ),
        0,
    )
    agg = (
        db.session.query(money_issued, money_returned, qty_issued, qty_returned)
        .select_from(IssuedPartRecord)
        .outerjoin(IssuedBatch, IssuedBatch.id == IssuedPartRecord.batch_id)
        .filter(base_q.whereclause if base_q.whereclause is not None else True)
        .one()
    )

    issued_total = float(agg[0] or 0.0)
    returned_total = float(agg[1] or 0.0)
    net_total = issued_total - returned_total
    issued_qty = int(agg[2] or 0)
    returned_qty = int(agg[3] or 0)
    net_qty = issued_qty - returned_qty

    # group issued items into "batches" for technician view
    def _fmt(dt):
        return dt.strftime("%Y-%m-%d %H:%M") if dt else "—"

    def _extract(rec):
        return (
            rec.id,
            (getattr(rec.part, "part_number", None) or "").strip().upper(),
            getattr(rec.part, "name", None) or "",
            int(rec.quantity or 0),
            float(rec.unit_cost_at_issue or 0.0),
            bool(getattr(rec, "confirmed_by_tech", False)),
            getattr(rec, "confirmed_by", None) or None,
            _fmt(getattr(rec, "confirmed_at", None)),
        )

    grouped = defaultdict(list)
    for r in issued_items:
        if getattr(r, "batch_id", None):
            key = ("batch", r.batch_id)
        elif getattr(r, "invoice_number", None):
            key = ("inv", r.invoice_number)
        else:
            key = (
                "ungrouped",
                f"{r.issue_date:%Y%m%d%H%M%S}-{r.issued_to}-{r.reference_job or ''}",
            )
        grouped[key].append(r)

    from collections import defaultdict as _dd
    net_by_pn = _dd(int)
    batches = []
    for _, recs in grouped.items():
        b = recs[0].batch if recs and getattr(recs[0], "batch", None) else None
        issued_at = (b.issue_date if b else recs[0].issue_date)
        issued_at_str = _fmt(issued_at)
        tech = recs[0].issued_to
        ref = (b.reference_job if b else (recs[0].reference_job or "")) or ""
        report_id = (b.invoice_number if b else recs[0].invoice_number)
        location = getattr(b, "location", None)

        ref_is_return = "RETURN" in ref.upper()

        items = []
        total_value_raw = 0.0
        unconfirmed_count = 0

        for rec in recs:
            rid, pn, name, qty, price, confirmed, who, when_s = _extract(rec)
            if not pn or qty == 0:
                continue
            line_total = qty * price
            total_value_raw += line_total

            is_item_return = ref_is_return or (qty < 0)
            eff_sign = -1 if is_item_return else 1
            net_by_pn[pn] += eff_sign * abs(qty)

            if (not confirmed) and qty > 0:
                unconfirmed_count += 1

            items.append(
                {
                    "id": rid,
                    "pn": pn,
                    "name": name or "—",
                    "qty": abs(qty),
                    "unit_price": price,
                    "negative": (line_total < 0) or is_item_return,
                    "confirmed": confirmed,
                    "confirmed_by": who,
                    "confirmed_at": when_s,
                }
            )

        is_group_return = (total_value_raw < 0) or ref_is_return
        all_conf = (unconfirmed_count == 0) and bool(items)
        any_conf = any(it.get("confirmed") for it in items) if items else False

        batches.append(
            {
                "issued_at": issued_at_str,
                "technician": tech,
                "canonical_ref": canon,
                "reference_job": ref,
                "location": location,
                "report_id": report_id,
                "invoice_number": report_id,
                "is_return": is_group_return,
                "total_value": total_value_raw,
                "items": items,
                "all_confirmed": all_conf,
                "any_confirmed": any_conf,
                "unconfirmed_count": unconfirmed_count,
            }
        )

    invoiced_pns = sorted([pn for pn, net in net_by_pn.items() if net > 0])

    # 5) Build blended pricing for PARTS TABLE (plan)
    # Gather all WorkOrderPart rows in display order
    all_parts = []
    if wo.units:
        for u in wo.units:
            if u.parts:
                all_parts.extend(u.parts)
    if (not all_parts) and wo.parts:
        all_parts.extend(wo.parts)

    # Map issued info by PN (sum qty & value with cost at issue)
    issued_info_by_pn = {}
    for rec in issued_items:
        pn = (getattr(rec.part, "part_number", "") or "").strip().upper()
        if not pn:
            continue
        q = int(rec.quantity or 0)
        cost_at_issue = float(rec.unit_cost_at_issue or 0.0)
        # contribution to value is q * that cost (can be negative for returns)
        val = q * cost_at_issue

        info = issued_info_by_pn.get(pn)
        if not info:
            info = {"qty": 0, "value": 0.0}
            issued_info_by_pn[pn] = info
        info["qty"] += q
        info["value"] += val

    # Fetch current inventory cost for all part_numbers in this WO
    pn_list = []
    for p in all_parts:
        pn = (p.part_number or "").strip().upper()
        if pn and pn not in pn_list:
            pn_list.append(pn)

    inv_cost_map = {}
    if pn_list:
        parts_rows = (
            db.session.query(Part)
            .filter(func.upper(Part.part_number).in_(pn_list))
            .all()
        )
        for pr in parts_rows:
            key = (pr.part_number or "").strip().upper()
            inv_cost_map[key] = float(pr.unit_cost or 0.0)

    # Build final display rows
    display_parts = []
    grand_total_display = 0.0

    for p in all_parts:
        pn = (p.part_number or "").strip().upper()
        qty_planned = int(p.quantity or 0)

        # issued side
        issued_qty_raw = 0
        issued_val_raw = 0.0
        ii = issued_info_by_pn.get(pn)
        if ii:
            issued_qty_raw = int(ii["qty"] or 0)
            issued_val_raw = float(ii["value"] or 0.0)

        # We only "count" issued if net qty > 0 after returns
        # If net <= 0, treat as 0 (everything returned)
        issued_qty_eff = issued_qty_raw if issued_qty_raw > 0 else 0
        issued_val_eff = issued_val_raw if issued_qty_raw > 0 else 0.0

        # remaining side
        not_issued_qty = qty_planned - issued_qty_eff
        if not_issued_qty < 0:
            not_issued_qty = 0

        current_inv_cost = inv_cost_map.get(pn, 0.0)
        not_issued_val = not_issued_qty * current_inv_cost

        blended_total_val = issued_val_eff + not_issued_val

        if qty_planned > 0:
            blended_unit_price = blended_total_val / qty_planned
        else:
            blended_unit_price = 0.0

        grand_total_display += blended_total_val

        # status flags for styling
        raw_oflag = getattr(p, "ordered_flag", None)
        ordered_flag_truthy = raw_oflag not in (None, False, 0, "0", "", "false", "False")
        is_ordered = (
            ordered_flag_truthy
            or (str(getattr(p, "status", "")).strip().lower() == "ordered")
            or (str(getattr(p, "line_status", "")).strip().lower() == "ordered")
        )
        is_invoiced = pn in invoiced_pns

        display_parts.append(
            {
                "part_number": p.part_number,
                "alt_part_numbers": getattr(p, "alt_part_numbers", "") or getattr(p, "alt_numbers", "") or "",
                "part_name": p.part_name or "—",
                "qty": qty_planned,
                "unit_price_display": blended_unit_price,
                "total_display": blended_total_val,
                "supplier": getattr(p, "supplier", "") or "",
                "is_ordered": is_ordered,
                "ordered_date": getattr(p, "ordered_date", None),
                "backorder_flag": bool(getattr(p, "backorder_flag", False)),
                "is_invoiced": is_invoiced,
            }
        )

    return render_template(
        "wo_detail.html",
        wo=wo,
        # blended view data for PARTS TABLE
        display_parts=display_parts,
        grand_total_display=grand_total_display,

        # for lower sections
        batches=batches,
        suppliers=suppliers,
        invoiced_pns=invoiced_pns,
        issued_items=issued_items,
        issued_total=issued_total,
        returned_total=returned_total,
        net_total=net_total,
        issued_qty=issued_qty,
        returned_qty=returned_qty,
        net_qty=net_qty,

        # perms
        is_my_wo=is_my_wo,
        can_confirm=can_confirm_any,       # legacy var
        can_confirm_any=can_confirm_any,   # superadmin only
        can_view_docs=can_view_docs,
    )

@inventory_bp.post("/work_orders/new", endpoint="wo_create")
@login_required
def wo_create():
    role = (getattr(current_user, "role", "") or "").strip().lower()
    if role not in ("admin", "superadmin"):
        flash("Access denied", "danger")
        return redirect(url_for("inventory.wo_list"))

    f = request.form

    # NEW: prefer technician_id; fallback to technician_name
    tech_id_raw = (f.get("technician_id") or "").strip()
    technician_id = int(tech_id_raw) if tech_id_raw.isdigit() else None
    technician_name = (f.get("technician_name") or "").strip()

    if technician_id:
        tech_obj = db.session.get(User, technician_id)
        if not tech_obj or (tech_obj.role or "").lower() != "technician":
            flash("Selected technician is invalid.", "danger")
            return redirect(url_for("inventory.wo_new"))
        technician_name = tech_obj.username  # normalize to username
    else:
        # require at least a name when no id provided
        if not technician_name:
            flash("Technician is required.", "danger")
            return redirect(url_for("inventory.wo_new"))

    job_numbers   = (f.get("job_numbers")   or "").strip()
    brand         = (f.get("brand")         or "").strip()
    model         = (f.get("model")         or "").strip()
    serial        = (f.get("serial")        or "").strip()
    job_type      = (f.get("job_type")      or "BASE").strip().upper()

    def _f(x, default=0.0):
        try: return float(x)
        except Exception: return float(default)

    delivery_fee   = _f(f.get("delivery_fee"), 0)
    markup_percent = _f(f.get("markup_percent"), 0)

    if not job_numbers:
        flash("Job(s) are required.", "danger")
        return redirect(url_for("inventory.wo_new"))

    # normalize job list
    jobs = [j.strip() for j in job_numbers.replace(",", " ").split() if j.strip()]

    # create WO
    wo = WorkOrder(
        technician_id=technician_id,       # NEW
        technician_name=technician_name,   # keep legacy compatibility
        job_numbers=", ".join(jobs),
        brand=brand, model=model, serial=serial,
        job_type=job_type if job_type in ("BASE", "INSURANCE") else "BASE",
        delivery_fee=delivery_fee,
        markup_percent=markup_percent,
        status="search_ordered",
    )
    db.session.add(wo)
    db.session.flush()

    # parse part rows
    import re
    pat = re.compile(r"^rows\[(\d+)\]\[(\w+)\]$")
    tmp = {}
    for k, v in f.items():
        m = pat.match(k)
        if not m: continue
        i = int(m.group(1)); field = m.group(2)
        tmp.setdefault(i, {})[field] = (v or "").strip()

    def _clip20(s: str) -> str: return (s or "").strip()[:20]
    def _clip6(s: str)  -> str: return (s or "").strip()[:6]

    for i in sorted(tmp.keys()):
        row = tmp[i]
        pn = _clip20((row.get("part_number") or "").upper())
        if not pn:
            continue

        alt_raw = (row.get("alt_numbers") or row.get("alt_part_numbers") or "")
        alt_tokens = [_clip20(t) for t in alt_raw.split(",") if t is not None]
        alt_joined = ",".join(alt_tokens)

        try:
            qty = int(row.get("quantity") or 0)
        except Exception:
            qty = 0

        part = WorkOrderPart(
            work_order_id=wo.id,
            part_number=pn,
            alt_numbers=alt_joined,
            part_name=(row.get("part_name") or "").strip(),
            quantity=qty,
            supplier=_clip6(row.get("supplier") or ""),
            backorder_flag=bool(row.get("backorder_flag")),
            status="search_ordered",
        )
        db.session.add(part)

    db.session.commit()
    flash("Work order created.", "success")
    return redirect(url_for("inventory.wo_detail", wo_id=wo.id))

def _render_new_wo_form(prefill=None, units_prefill=None, flash_msg=None):
    """
    Рендер той самой multi-appliance формы /work_orders/new
    без редиректа, с уже введёнными данными.
    """
    from flask import session, render_template, flash
    from flask_login import current_user
    from models import User

    # recent suppliers (чтоб datalist и т.п. остались как раньше)
    recent_suppliers = session.get("recent_suppliers", [])

    if flash_msg:
        flash(flash_msg, "warning")

    # ---------- technician dropdown ----------
    # у тебя в шаблоне "This list includes only users with the technician role."
    tech_users = User.query.filter(
        (User.role == "technician") | (User.role == "TECHNICIAN")
    ).order_by(User.username.asc()).all()

    # ---------- шапка ордера ----------
    # prefill может быть WorkOrder (несохранённый) или dict
    wo_hdr = {
        "technician_id":    "",
        "technician_name":  "",
        "job_numbers":      "",
        "job_type":         "BASE",
        "delivery_fee":     0.0,
        "markup_percent":   0.0,
        "status":           "search_ordered",
    }
    if prefill:
        # аккуратно копируем поля, если есть
        wo_hdr["technician_id"]   = getattr(prefill, "technician_id",   "") or ""
        wo_hdr["technician_name"] = getattr(prefill, "technician_name", "") or ""
        wo_hdr["job_numbers"]     = getattr(prefill, "job_numbers",     "") or ""
        wo_hdr["job_type"]        = getattr(prefill, "job_type",        "BASE") or "BASE"
        wo_hdr["delivery_fee"]    = getattr(prefill, "delivery_fee",    0.0) or 0.0
        wo_hdr["markup_percent"]  = getattr(prefill, "markup_percent",  0.0) or 0.0
        wo_hdr["status"]          = getattr(prefill, "status",          "search_ordered") or "search_ordered"

    # ---------- блоки appliances / units / rows ----------
    # Это то, что ты собираешь внутри wo_save в units_payload.
    # Если у нас уже есть units_prefill (спарсили из формы перед валидацией),
    # то отдаём его; иначе отдаём дефолт с одним appliance и одной пустой строкой.
    if units_prefill:
        units_for_template = units_prefill
    else:
        units_for_template = [{
            "brand": "",
            "model": "",
            "serial": "",
            "rows": [{
                "part_number": "",
                "part_name": "",
                "quantity": 1,
                "alt_numbers": "",
                "warehouse": "",
                "supplier": "",
                "backorder_flag": False,
                "ordered_flag": False,
                "unit_cost": "",
                "line_status": "search_ordered",
                "stock_hint": "",
            }],
        }]

    # ВАЖНО: этот шаблон должен быть ТОЧНО той страницей, которая у тебя на /work_orders/new сейчас.
    # Если он у тебя называется иначе – подставь правильный путь.
    return render_template(
        "work_orders/new.html",
        wo=wo_hdr,
        units=units_for_template,
        tech_users=tech_users,
        recent_suppliers=recent_suppliers,
        readonly=False,
    )

@inventory_bp.get("/work_orders/newx", endpoint="wo_newx")
@login_required
def wo_newx(prefill_wo=None, prefill_units=None):
    from flask import session, render_template
    recent_suppliers = session.get("recent_suppliers", [])

    role_low = (getattr(current_user, "role", "") or "").lower()
    if role_low not in ("admin", "superadmin"):
        flash("Access denied", "danger")
        return redirect(url_for("inventory.wo_list"))

    if prefill_wo is not None:
        wo = prefill_wo
    else:
        wo = None

    if prefill_units is not None:
        units = prefill_units
    else:
        units = [{
            "brand": "",
            "model": "",
            "serial": "",
            "rows": [{
                "part_number": "",
                "part_name": "",
                "quantity": 1,
                "alt_numbers": "",
                "supplier": "",
                "backorder_flag": False,
                "line_status": "search_ordered",
                "warehouse": "",
                "unit_cost": "",
            }],
        }]

    return render_template(
        "wo_form.html",
        wo=wo,
        units=units,
        readonly=False,
        recent_suppliers=recent_suppliers,
    )

@inventory_bp.post("/work_orders/save", endpoint="wo_save")
@login_required
def wo_save():
    from flask import request, session, redirect, url_for, flash, current_app, render_template
    from models import WorkOrder, WorkUnit, WorkOrderPart, User
    from extensions import db
    from flask_login import current_user
    import re
    from datetime import date

    # ---------- helpers ----------
    def _clip(s, n):
        return (s or "").strip()[:n]

    def _i(x, default=0):
        try:
            return int(x)
        except Exception:
            return int(default)

    def _f(x, default=None):
        if x is None or x == "":
            return default
        try:
            return float(x)
        except Exception:
            return default

    def _b(x):
        return str(x).strip().lower() in ("1", "true", "on", "yes", "y")

    def _safe_detail_redirect(work_order):
        wo_id_val = getattr(work_order, "id", None)
        if wo_id_val:
            return redirect(url_for("inventory.wo_detail", wo_id=wo_id_val))
        return redirect(url_for("inventory.wo_list"))

    # ---------- access control ----------
    role_low = (getattr(current_user, "role", "") or "").strip().lower()
    if role_low not in ("admin", "superadmin"):
        flash("Access denied", "danger")
        return redirect(url_for("inventory.wo_list"))

    f = request.form
    log_keys = sorted(f.keys())
    current_app.logger.debug("WO_SAVE keys (%s): %s", len(log_keys), log_keys[:200])

    # new vs edit
    wo_id  = (f.get("wo_id") or "").strip()
    is_new = not wo_id

    # взять WorkOrder
    if is_new:
        wo = WorkOrder()
    else:
        wo = WorkOrder.query.get_or_404(int(wo_id))

    # ---------- заголовок / шапка ----------
    tech_id_raw   = (f.get("technician_id")   or f.get("technician") or "").strip()
    tech_name_raw = (f.get("technician_name") or f.get("technician") or "").strip()

    tech_id_val   = None
    tech_name_val = None

    # если пришёл ID техника — пробуем найти пользователя
    if tech_id_raw.isdigit():
        try:
            tid = int(tech_id_raw)
            u = User.query.get(tid)
            if u:
                tech_id_val   = tid
                tech_name_val = (u.username or "").strip().upper()
        except Exception:
            pass

    # иначе используем введённое имя техника
    if not tech_name_val:
        if tech_name_raw:
            tech_name_val = tech_name_raw.strip().upper()
        else:
            tech_name_val = None

    wo.technician_id   = tech_id_val
    wo.technician_name = (tech_name_val or "").strip().upper() if tech_name_val else ""

    wo.job_numbers     = (f.get("job_numbers") or "").strip()
    wo.job_type        = (f.get("job_type") or "BASE").strip().upper()
    wo.delivery_fee    = _f(f.get("delivery_fee"), 0) or 0.0
    wo.markup_percent  = _f(f.get("markup_percent"), 0) or 0.0

    st_field = (f.get("status") or "search_ordered").strip()
    wo.status = st_field if st_field in ("search_ordered", "ordered", "done") else "search_ordered"

    brand_hdr  = (f.get("brand")  or "").strip()
    model_hdr  = _clip(f.get("model"), 25)
    serial_hdr = _clip(f.get("serial"), 25)
    if brand_hdr:
        wo.brand = brand_hdr
    if model_hdr:
        wo.model = model_hdr
    if serial_hdr:
        wo.serial = serial_hdr

    # ---------- собрать units[...] и их rows[...] ----------
    re_unit = re.compile(r"^units\[(\d+)\]\[(brand|model|serial)\]$")
    re_row  = re.compile(
        r"^units\[(\d+)\]\[rows\]\[(\d+)\]\[(part_number|part_name|quantity|"
        r"alt_numbers|alt_part_numbers|warehouse|unit_label|supplier|supplier_name|"
        r"backorder_flag|status|unit_cost|ordered_flag)\]$"
    )

    units_map = {}
    for key in f.keys():
        m = re_unit.match(key)
        if m:
            ui, name = int(m.group(1)), m.group(2)
            units_map.setdefault(ui, {"rows": {}})
            units_map[ui][name] = f.get(key)
            continue
        m = re_row.match(key)
        if m:
            ui, ri, name = int(m.group(1)), int(m.group(2)), m.group(3)
            units_map.setdefault(ui, {"rows": {}})
            units_map[ui]["rows"].setdefault(ri, {})
            # чекбоксы могут дублироваться: берём последний
            if name in ("ordered_flag", "backorder_flag"):
                vals = request.form.getlist(key)
                val  = vals[-1] if vals else f.get(key)
            else:
                val  = f.get(key)
            units_map[ui]["rows"][ri][name] = val

    # превратить units_map -> units_payload
    units_payload  = []
    new_rows_count = 0

    for ui in sorted(units_map.keys()):
        u_blk = units_map[ui]
        rows_payload = []

        for ri in sorted(u_blk.get("rows", {}).keys()):
            r = u_blk["rows"][ri]

            pn  = (r.get("part_number") or "").strip().upper()
            qty = _i(r.get("quantity") or 0, 0)

            alt_raw  = (r.get("alt_numbers") or r.get("alt_part_numbers") or "").strip()
            wh_raw   = (r.get("warehouse")  or r.get("unit_label") or "").strip()
            sup_raw  = (r.get("supplier")   or r.get("supplier_name") or "").strip()

            ucost    = _f(r.get("unit_cost"), None)
            bo_flag  = _b(r.get("backorder_flag"))
            lstatus  = (r.get("status") or "search_ordered").strip()
            ord_flag = _b(r.get("ordered_flag"))

            row_dict = {
                "id": None,
                "part_number":    _clip(pn, 80),
                "part_name":      _clip(r.get("part_name"), 120),
                "quantity":       qty if qty else 1,
                "alt_numbers":    _clip(alt_raw, 200),
                "warehouse":      _clip(wh_raw, 120),
                "supplier":       _clip(sup_raw, 80),
                "backorder_flag": bo_flag,
                "line_status":    lstatus if lstatus in ("search_ordered", "ordered", "done") else "search_ordered",
                "unit_cost":      (ucost if (ucost is not None) else 0.0),
                "ordered_flag":   ord_flag,   # <-- сохраняем!
            }

            if pn and qty > 0:
                new_rows_count += 1

            rows_payload.append(row_dict)

        units_payload.append({
            "brand":  (u_blk.get("brand")  or "").strip(),
            "model":  _clip(u_blk.get("model"), 25),
            "serial": _clip(u_blk.get("serial"), 25),
            "rows":   rows_payload,
        })

    current_app.logger.debug(
        "WO_SAVE parsed units=%s rows_total=%s",
        len(units_payload),
        sum(len(u.get('rows') or []) for u in units_payload)
    )

    # DEBUG: показать что у нас получилось по ordered_flag до сохранения
    try:
        dbg_rows = []
        for u_i, u_blk in sorted(units_map.items()):
            for r_i, r_blk in sorted((u_blk.get("rows") or {}).items()):
                dbg_rows.append({
                    "u": u_i,
                    "r": r_i,
                    "pn": (r_blk.get("part_number") or "").strip().upper(),
                    "ord_raw": r_blk.get("ordered_flag"),
                    "bo_raw": r_blk.get("backorder_flag"),
                    "status": r_blk.get("status"),
                })
        current_app.logger.debug("WO_SAVE DEBUG rows before validation: %s", dbg_rows)
    except Exception as _e:
        current_app.logger.debug("WO_SAVE DEBUG build dbg_rows failed: %r", _e)

    # ---------- ререндер формы при ошибке ----------
    def _rerender_same_screen(msg_text: str):
        db.session.rollback()
        flash(msg_text, "warning")

        technicians = _query_technicians()
        recent_suppliers = session.get("recent_suppliers", []) or []

        sel_tid   = wo.technician_id
        sel_tname = wo.technician_name or None

        safe_units = units_payload if units_payload else [{
            "brand":  wo.brand or "",
            "model":  wo.model or "",
            "serial": wo.serial or "",
            "rows": [{
                "id": None,
                "part_number": "",
                "part_name": "",
                "quantity": 1,
                "alt_numbers": "",
                "warehouse": "",
                "supplier": "",
                "backorder_flag": False,
                "line_status": "search_ordered",
                "unit_cost": 0.0,
                "ordered_flag": False,
            }],
        }]

        return render_template(
            "wo_form_units.html",
            wo=wo,
            units=safe_units,
            recent_suppliers=recent_suppliers,
            readonly=False,
            technicians=technicians,
            selected_tech_id=sel_tid,
            selected_tech_username=sel_tname,
        )

    # ---------- ВАЛИДАЦИЯ ----------
    if not tech_name_val:
        if is_new:
            return _rerender_same_screen("Technician is required before saving Work Order.")
        else:
            db.session.rollback()
            flash("Technician is required before saving Work Order.", "warning")
            return redirect(url_for("inventory.wo_detail", wo_id=wo.id))

    if not wo.job_numbers:
        if is_new:
            return _rerender_same_screen("Job number is required.")
        else:
            db.session.rollback()
            flash("Job number is required.", "warning")
            return redirect(url_for("inventory.wo_detail", wo_id=wo.id))

    if is_new:
        db.session.add(wo)

    # если шапка без строк
    if new_rows_count == 0:
        if units_payload:
            first = units_payload[0]
            wo.brand  = (first.get("brand")  or "").strip() or None
            wo.model  = _clip(first.get("model"), 25) or None
            wo.serial = _clip(first.get("serial"), 25) or None

        try:
            db.session.commit()
            flash("Work Order saved.", "success")
        except Exception as e:
            db.session.rollback()
            flash(f"Failed to save Work Order: {e}", "danger")
            return redirect(url_for("inventory.wo_list"))

        return redirect(url_for("inventory.wo_detail", wo_id=wo.id))

    if not any(u.get("rows") for u in units_payload):
        flash("Nothing parsable in rows. Nothing was changed.", "warning")
        db.session.rollback()
        return _safe_detail_redirect(wo)

    # ---------- сохранить прежние ordered_flag / ordered_date ----------
    def _norm_supplier(s):
        s = (s or "").strip()
        return " ".join(s.split()).lower()

    def _unit_key(b, m, s):
        return (
            _clip(b, 80).lower(),
            _clip(m, 25).lower(),
            _clip(s, 25).lower()
        )

    old_index = {}
    if not is_new:
        for old_u in (wo.units or []):
            uk = _unit_key(old_u.brand or "", old_u.model or "", old_u.serial or "")
            for old_p in (old_u.parts or []):
                pn   = ((old_p.part_number or "").strip().upper())
                supn = _norm_supplier(old_p.supplier)
                was_ordered = (
                    bool(getattr(old_p, "ordered_flag", False)) or
                    ((getattr(old_p, "status", "") or "").lower() == "ordered") or
                    ((getattr(old_p, "line_status", "") or "").lower() == "ordered")
                )
                old_index[(uk, pn, supn)] = (
                    was_ordered,
                    getattr(old_p, "ordered_date", None)
                )

    # ---------- удалить старые юниты/parts и пересоздать ----------
    for u in list(getattr(wo, "units", []) or []):
        for p in list(getattr(u, "parts", []) or []):
            db.session.delete(p)
        db.session.delete(u)

    try:
        db.session.flush()
    except Exception as e:
        db.session.rollback()
        flash(f"Failed to refresh units on Work Order: {e}", "danger")
        return _safe_detail_redirect(wo)

    # синхронизируем заголовок WO с первым юнитом
    if units_payload:
        first_unit = units_payload[0]
        wo.brand  = (first_unit.get("brand")  or "").strip() or None
        wo.model  = _clip(first_unit.get("model"), 25) or None
        wo.serial = _clip(first_unit.get("serial"), 25) or None

    suppliers_seen = []

    # ---------- создаём заново WorkUnit и WorkOrderPart ----------
    for up in units_payload:
        unit = WorkUnit(
            work_order=wo,
            brand=(up.get("brand") or "").strip(),
            model=_clip(up.get("model"), 25),
            serial=_clip(up.get("serial"), 25),
        )
        db.session.add(unit)

        try:
            db.session.flush()
        except Exception as e:
            db.session.rollback()
            flash(f"Failed to add unit: {e}", "danger")
            return _safe_detail_redirect(wo)

        uk = _unit_key(unit.brand or "", unit.model or "", unit.serial or "")

        for r in (up.get("rows") or []):
            sup = r.get("supplier") or ""
            if sup:
                norm = " ".join(sup.split())
                if norm and norm.lower() not in [x.lower() for x in suppliers_seen]:
                    suppliers_seen.append(norm)

            pn_upper = (r.get("part_number") or "").strip().upper()
            sup_norm = _norm_supplier(sup)

            # что пришло из формы по "ordered_flag"
            ord_in_raw = r.get("ordered_flag")
            ord_in     = _b(ord_in_raw)

            prev_state = old_index.get((uk, pn_upper, sup_norm))
            prev_was_ordered, prev_date = (prev_state if prev_state else (False, None))

            if ord_in:
                # если раньше уже был ordered — оставляем ту же дату
                if prev_was_ordered and prev_date:
                    ord_date = prev_date
                else:
                    ord_date = date.today()
            else:
                # галку не поставили → не ordered
                ord_date = None
                ord_in   = False

            current_app.logger.debug(
                "WO_SAVE DEBUG build part row pn=%s sup=%s ord_in_raw=%r -> ord_in=%r prev_was_ordered=%r prev_date=%r final_date=%r",
                pn_upper, sup_norm, ord_in_raw, ord_in, prev_was_ordered, prev_date, ord_date
            )

            wop = WorkOrderPart(
                work_order=wo,
                unit=unit,
                part_number=pn_upper,
                part_name=(r.get("part_name") or None),
                quantity=int(r.get("quantity") or 0),
                alt_part_numbers=(r.get("alt_numbers") or None),
                supplier=(sup or None),
                backorder_flag=bool(r.get("backorder_flag")),
                status=("ordered" if ord_in else "search_ordered"),
            )

            # склад
            if hasattr(wop, "warehouse"):
                wop.warehouse = (r.get("warehouse") or "")[:120]
            wop.unit_label = (r.get("warehouse") or "")[:120] or None

            # цена
            if hasattr(wop, "unit_cost"):
                uc = r.get("unit_cost")
                if uc is not None and uc != "":
                    try:
                        wop.unit_cost = float(uc)
                    except Exception:
                        wop.unit_cost = None

            # ordered info
            if hasattr(wop, "ordered_flag"):
                wop.ordered_flag = ord_in
            if hasattr(wop, "ordered_date"):
                wop.ordered_date = ord_date
            if hasattr(wop, "line_status"):
                wop.line_status = "ordered" if ord_in else "search_ordered"

            db.session.add(wop)

    # ---------- commit ----------
    try:
        db.session.commit()
        flash("Work Order saved.", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"Failed to save Work Order: {e}", "danger")
        return redirect(url_for("inventory.wo_list"))

    # ---------- remember suppliers ----------
    if suppliers_seen:
        cur = session.get("recent_suppliers", [])
        merged, seen = [], set()
        for x in suppliers_seen + list(cur):
            xl = x.lower()
            if xl in seen:
                continue
            seen.add(xl)
            merged.append(x)
        session["recent_suppliers"] = merged[:20]
        session.modified = True

    return redirect(url_for("inventory.wo_detail", wo_id=wo.id))

@inventory_bp.get("/api/stock_hint", endpoint="api_stock_hint")
@login_required
def api_stock_hint():
    from flask import request, jsonify
    from sqlalchemy import func
    from models import Part  # подстрой импорт, если у тебя другой путь

    # --- вход ---
    pn = (request.args.get("pn") or "").strip().upper()
    try:
        qty = int(request.args.get("qty") or 0)
    except Exception:
        qty = 0

    wh = (request.args.get("wh") or request.args.get("warehouse") or "").strip()

    if not pn or qty <= 0:
        return jsonify({"hint": "—", "available_qty": 0})

    # --- хелперы ---
    def pick_int(obj, names, default=0):
        for n in names:
            if hasattr(obj, n):
                v = getattr(obj, n)
                if v is None:
                    continue
                try:
                    return int(v)
                except Exception:
                    try:
                        return int(float(v))
                    except Exception:
                        continue
        return int(default)

    def pick_num(obj, names):
        for n in names:
            if hasattr(obj, n):
                v = getattr(obj, n)
                if v is None:
                    continue
                try:
                    return float(v)
                except Exception:
                    try:
                        return float(str(v).strip().replace(",", ""))
                    except Exception:
                        continue
        return None

    def pick_str(obj, names):
        for n in names:
            if hasattr(obj, n):
                v = getattr(obj, n)
                if v is None:
                    continue
                s = str(v).strip()
                if s:
                    return s
        return None

    # --- поиск Part по PN (+опционально по складу) ---
    q = Part.query.filter(func.upper(Part.part_number) == pn)

    if wh and any(hasattr(Part, n) for n in ("location", "warehouse", "wh")):
        if hasattr(Part, "location"):
            q = q.filter(Part.location == wh)
        elif hasattr(Part, "warehouse"):
            q = q.filter(Part.warehouse == wh)
        elif hasattr(Part, "wh"):
            q = q.filter(Part.wh == wh)

    part = q.first()
    if not part:
        # отдадим минимально полезный ответ, фронт покажет WAIT и не будет автозаполнять
        return jsonify({
            "hint": "WAIT: unknown PN",
            "available_qty": 0,
            "on_hand": 0,
            "ordered": 0,
            "part_name": "",
            "unit_cost": None,
            "warehouse": wh or "",
            "pn": pn,
            "requested_qty": qty
        })

    # --- запасы/заказы/ETA ---
    on_hand = pick_int(part, ("on_hand", "quantity", "qty_on_hand", "stock", "in_stock"), default=0)
    ordered = pick_int(part, ("ordered_qty", "on_order", "ordered", "po_qty"), default=0)
    eta_val = None
    for n in ("eta", "expected_date", "arrival_date", "due_date"):
        if hasattr(part, n):
            eta_val = getattr(part, n)
            break

    # --- автозаполнение: имя, цена, склад ---
    part_name = pick_str(part, ("part_name", "name", "title", "description")) or ""
    unit_cost = pick_num(part, ("unit_cost", "price", "unit_price", "cost", "last_price", "avg_cost"))
    if unit_cost is not None:
        unit_cost = round(unit_cost, 2)
    warehouse = pick_str(part, ("warehouse", "location", "wh")) or (wh or "")

    # --- хинт ---
    if on_hand >= qty:
        hint = "STOCK"
    else:
        hint = "WAIT"

    # --- ответ фронту (важно: ключи available_qty, part_name, unit_cost, warehouse) ---
    payload = {
        "hint": hint,
        "available_qty": on_hand,   # фронт читает это поле
        "on_hand": on_hand,
        "ordered": ordered,
        "eta": (eta_val.isoformat() if hasattr(eta_val, "isoformat") else (str(eta_val).strip() if eta_val else None)),
        "location": warehouse or None,

        # ключевое для автоподстановки:
        "part_name": part_name,
        "unit_cost": unit_cost,
        "warehouse": warehouse,

        "pn": pn,
        "requested_qty": qty,
    }
    return jsonify(payload)

@inventory_bp.get("/work_orders")
@login_required
def wo_list():
    """
    Work Orders list with optional filters:
    - tech: technician_name ILIKE
    - jobs: job_numbers ILIKE
    - model: brand OR model ILIKE
    - pn: part_number OR alt_part_numbers ILIKE (outer join)
    - from/to: created_at date range (inclusive)

    Дополнительно:
    - если current_user.role == 'technician' → показываем только его WO
      (по technician_id == current_user.id, с fallback по точному совпадению имени)
    """
    tech  = (request.args.get("tech") or "").strip()
    jobs  = (request.args.get("jobs") or "").strip()
    model = (request.args.get("model") or "").strip()
    pn    = (request.args.get("pn") or "").strip()
    dfrom = (request.args.get("from") or "").strip()  # YYYY-MM-DD
    dto   = (request.args.get("to") or "").strip()    # YYYY-MM-DD

    q = db.session.query(WorkOrder)
    joined_parts = False
    filters = []

    # --- ОГРАНИЧЕНИЕ ДЛЯ ТЕХНИКА ---
    if is_technician():
        me_id = getattr(current_user, "id", None)
        me_name = (getattr(current_user, "username", "") or "").strip()
        # technician_id — основной фильтр; technician_name — fallback для старых записей
        filters.append(
            or_(
                WorkOrder.technician_id == me_id,
                func.trim(WorkOrder.technician_name) == me_name,
            )
        )

    # --- Остальные твои фильтры (как было) ---
    if tech:
        filters.append(WorkOrder.technician_name.ilike(f"%{tech}%"))

    if jobs:
        filters.append(WorkOrder.job_numbers.ilike(f"%{jobs}%"))

    if model:
        filters.append(or_(
            WorkOrder.brand.ilike(f"%{model}%"),
            WorkOrder.model.ilike(f"%{model}%"),
        ))

    if pn:
        # Подключаем части только если реально фильтруем по ним
        q = q.outerjoin(WorkOrderPart, WorkOrderPart.work_order_id == WorkOrder.id)
        joined_parts = True
        filters.append(or_(
            WorkOrderPart.part_number.ilike(f"%{pn}%"),
            WorkOrderPart.alt_part_numbers.ilike(f"%{pn}%"),
        ))

    # Диапазон дат по created_at (включительно)
    if dfrom:
        try:
            start = datetime.strptime(dfrom, "%Y-%m-%d")
            filters.append(WorkOrder.created_at >= start)
        except ValueError:
            pass

    if dto:
        try:
            end = datetime.strptime(dto, "%Y-%m-%d") + timedelta(days=1) - timedelta(seconds=1)
            filters.append(WorkOrder.created_at <= end)
        except ValueError:
            pass

    if filters:
        q = q.filter(and_(*filters))

    if joined_parts:
        # чтобы не дублировать WO при outer join
        q = q.distinct(WorkOrder.id)

    items = q.order_by(WorkOrder.created_at.desc()).limit(200).all()
    return render_template("wo_list.html", items=items, args=request.args)

@inventory_bp.post("/work_orders/<int:wo_id>/issue_instock", endpoint="wo_issue_instock")
@login_required
def wo_issue_instock(wo_id):
    if getattr(current_user, "role", "") not in ("admin", "superadmin"):
        flash("Access denied", "danger")
        return redirect(url_for("inventory.wo_detail", wo_id=wo_id))

    from urllib.parse import urlencode
    from markupsafe import Markup
    from sqlalchemy import func
    from datetime import datetime
    from flask import request, session

    from extensions import db
    from models import WorkOrder, WorkOrderPart, Part  # Part — складская позиция

    wo = WorkOrder.query.get_or_404(wo_id)

    # ===== флаг автосмены статуса (done или нет) =====
    set_status = (request.form.get("set_status") or "").strip().lower()

    # === 1) расчёт доступности (для greedy-проверки) ===
    try:
        avail_rows = compute_availability(wo) or []
    except Exception:
        avail_rows = []

    stock_map = {}  # PN -> on_hand (по расчёту)
    hint_map  = {}  # PN -> hint/status
    for r in avail_rows:
        pn = (r.get("part_number") or "").strip().upper()
        if not pn:
            continue
        on_hand = int(r.get("on_hand") or 0)
        stock_map[pn] = stock_map.get(pn, 0) + on_hand
        hint_map[pn]  = (r.get("status_hint") or r.get("hint") or ("STOCK" if on_hand > 0 else "WAIT"))

    def can_issue(pn: str, qty: int) -> bool:
        pn = (pn or "").strip().upper()
        if not pn or qty <= 0:
            return False
        left = int(stock_map.get(pn, 0))
        if left >= qty:
            stock_map[pn] = left - qty
            return True
        return False

    # === 2) Режим “выбрано” — выдаём конкретные строки по их ID ===
    raw_ids = (request.form.get("part_ids") or "").strip()
    items_to_issue = []
    issued_row_ids = []
    skipped_rows = []

    if raw_ids:
        ids = [int(tok) for tok in raw_ids.split(",") if tok.strip().isdigit()]
        if not ids:
            flash("Nothing selected to issue.", "warning")
            return redirect(url_for("inventory.wo_detail", wo_id=wo_id))

        wops: list[WorkOrderPart] = (
            WorkOrderPart.query
            .filter(
                WorkOrderPart.work_order_id == wo_id,
                WorkOrderPart.id.in_(ids)
            )
            .all()
        )
        if not wops:
            flash("Selected parts not found.", "warning")
            return redirect(url_for("inventory.wo_detail", wo_id=wo_id))

        part_has_location = hasattr(Part, "location")

        for line in wops:
            pn = (line.part_number or "").strip().upper()
            qty_req = int(line.quantity or 0)
            if not pn or qty_req <= 0:
                continue

            # найдём конкретный складской Part
            q_base = Part.query.filter(func.upper(Part.part_number) == pn)
            part = None
            if part_has_location and getattr(line, "warehouse", None):
                part = q_base.filter(
                    func.coalesce(Part.location, "") == (line.warehouse or "")
                ).first()
            if not part:
                part = q_base.first()

            ok = can_issue(pn, qty_req)

            # доп. проверка: если greedy сказал "нет",
            # но фактически по складу qty хватает — разрешаем
            if not ok and part:
                try:
                    real_left = int(getattr(part, "quantity", 0) or 0)
                except Exception:
                    real_left = 0
                if real_left >= qty_req:
                    stock_map[pn] = int(stock_map.get(pn, 0))
                    stock_map[pn] = max(0, stock_map[pn] - qty_req)
                    ok = True

            hint_norm = (hint_map.get(pn) or ("STOCK" if ok else "WAIT"))
            if hasattr(line, "stock_hint"):
                try:
                    line.stock_hint = hint_norm
                except Exception:
                    pass

            if not part or not ok:
                # не смогли выдать эту строку
                skipped_rows.append({
                    "id": line.id,
                    "pn": pn,
                    "name": getattr(line, "part_name", "") or "—",
                    "qty": qty_req,
                    "hint": hint_norm
                })
                continue

            # ✅ ВАЖНО:
            # Фиксируем ЦЕНУ ДЛЯ ИНВОЙСА с реального склада, а не из WorkOrderPart.
            try:
                real_cost = float(part.unit_cost or 0.0)
            except Exception:
                real_cost = 0.0

            items_to_issue.append({
                "part_id": part.id,
                "qty": qty_req,
                "unit_price": real_cost,   # <-- вот это уйдёт в unit_cost_at_issue
            })
            issued_row_ids.append(line.id)

        if not items_to_issue:
            if skipped_rows:
                session["wo_skip_info"] = skipped_rows
                flash("Nothing available to issue (selected lines are not in stock).", "warning")
            else:
                flash("Nothing available to issue.", "warning")
            return redirect(url_for("inventory.wo_detail", wo_id=wo_id))

        # --- создаём записи выдачи ---
        issue_result = _issue_records_bulk(
            issued_to=wo.technician_name,
            reference_job=wo.canonical_job,
            items=items_to_issue
        )

        # поддержка обеих сигнатур: (dt, records) ИЛИ (dt, created_bool)
        issue_date = None
        new_records = None
        try:
            issue_date, new_records = issue_result  # новая сигнатура
        except Exception:
            issue_date, _created = issue_result     # старая сигнатура

        # отметим, сколько реально выдали в WorkOrderPart
        now = datetime.utcnow()
        for line in WorkOrderPart.query.filter(WorkOrderPart.id.in_(issued_row_ids)).all():
            qty_needed = int(line.quantity or 0)
            issued_so_far = int(getattr(line, "issued_qty", 0) or 0)
            issue_now = max(qty_needed - issued_so_far, 0)

            if issue_now > 0:
                line.issued_qty = issued_so_far + issue_now
                if line.issued_qty >= qty_needed:
                    try:
                        line.status = "done"
                    except Exception:
                        pass

            if hasattr(line, "last_issued_at"):
                line.last_issued_at = now

            db.session.add(line)
        db.session.commit()

        # автосмена статуса WO -> done (если попросили)
        if set_status == "done":
            try:
                wo.status = "done"
                db.session.commit()
            except Exception:
                db.session.rollback()

        # --- если есть новые записи - делаем инвойс и редиректим на него ---
        if new_records:
            try:
                batch = _ensure_invoice_number_for_records(
                    records=new_records,
                    issued_to=wo.technician_name,
                    issued_by=getattr(current_user, "username", "system"),
                    reference_job=wo.canonical_job,
                    issue_date=issue_date or datetime.utcnow(),
                    location=None
                )
                db.session.commit()
                return redirect(f"/reports/{batch.invoice_number}", code=303)
            except Exception:
                db.session.rollback()
                # если не удалось - пойдём в общий отчёт ниже

        # --- fallback: сгруппированный отчёт за день ---
        d = (issue_date or datetime.utcnow()).date().isoformat()
        params = urlencode({
            "start_date": d,
            "end_date": d,
            "recipient": wo.technician_name,
            "reference_job": wo.canonical_job
        })
        link = f"/reports_grouped?{params}"

        if skipped_rows:
            session["wo_skip_info"] = skipped_rows

        flash(Markup(
            f'Issued {len(issued_row_ids)} line(s). '
            f'<a href="{link}" target="_blank" rel="noopener">Open invoice group</a> to print.'
        ), "success")

        return redirect(url_for(
            "inventory.wo_detail",
            wo_id=wo.id,
            issued_ids=",".join(map(str, issued_row_ids))
        ))

    # === 3) Режим “ничего не выбрано” — массовая выдача in-stock ===
    # строим pn_issue_map и items_to_issue на основе avail_rows
    pn_issue_map = {}
    items_to_issue.clear()

    for r in avail_rows:
        issue_now = int(r.get("issue_now") or 0)
        if issue_now <= 0:
            continue

        pn = (r.get("part_number") or "").strip().upper()
        if not pn:
            continue

        pn_issue_map[pn] = pn_issue_map.get(pn, 0) + issue_now

        part = Part.query.filter(func.upper(Part.part_number) == pn).first()
        if not part:
            continue

        # ✅ тоже фиксируем реальную себестоимость склада
        try:
            real_cost = float(part.unit_cost or 0.0)
        except Exception:
            real_cost = 0.0

        items_to_issue.append({
            "part_id": part.id,
            "qty": issue_now,
            "unit_price": real_cost,  # фиксируем правильную цену
        })

    if not items_to_issue:
        flash("Nothing available to issue (all WAIT).", "warning")
        return redirect(url_for("inventory.wo_detail", wo_id=wo_id))

    issue_date, _created_or_records = _issue_records_bulk(
        issued_to=wo.technician_name,
        reference_job=wo.canonical_job,
        items=items_to_issue
    )

    # распределяем выданное количество по строкам этого WO (issued_qty / status / last_issued_at)
    now = datetime.utcnow()
    for pn, need_to_apply in pn_issue_map.items():
        if need_to_apply <= 0:
            continue

        rows = (
            WorkOrderPart.query
            .filter(
                WorkOrderPart.work_order_id == wo.id,
                func.upper(WorkOrderPart.part_number) == pn
            )
            .order_by(WorkOrderPart.id.asc())
            .all()
        )

        for line in rows:
            if need_to_apply <= 0:
                break

            qty_needed = int(line.quantity or 0)
            issued_so_far = int(getattr(line, "issued_qty", 0) or 0)
            remaining = max(qty_needed - issued_so_far, 0)
            if remaining <= 0:
                continue

            delta = min(remaining, need_to_apply)
            line.issued_qty = issued_so_far + delta

            if line.issued_qty >= qty_needed:
                try:
                    line.status = "done"
                except Exception:
                    pass

            if hasattr(line, "last_issued_at"):
                line.last_issued_at = now

            db.session.add(line)
            need_to_apply -= delta

    db.session.commit()

    # статус WO -> done, если попросили
    if set_status == "done":
        try:
            wo.status = "done"
            db.session.commit()
        except Exception:
            db.session.rollback()
    else:
        # иначе попробуем автоматически закрыть если ничего не ждём
        try:
            still_wait = any(
                (row.get("on_hand", 0) < row.get("requested", 0))
                for row in (compute_availability(wo) or [])
            )
        except Exception:
            still_wait = True

        if not still_wait:
            try:
                wo.status = "done"
                db.session.commit()
            except Exception:
                db.session.rollback()

    # сгруппированный отчёт за день
    d = (issue_date or datetime.utcnow()).date().isoformat()
    params = urlencode({
        "start_date": d,
        "end_date": d,
        "recipient": wo.technician_name,
        "reference_job": wo.canonical_job
    })
    link = f"/reports_grouped?{params}"

    flash(Markup(
        f'Issued in-stock items. '
        f'<a href="{link}" target="_blank" rel="noopener">Open invoice group</a> to print.'
    ), "success")

    return redirect(url_for("inventory.wo_detail", wo_id=wo_id))

@inventory_bp.get("/work_orders/<int:wo_id>/edit", endpoint="wo_edit")
@login_required
def wo_edit(wo_id: int):
    role = (getattr(current_user, "role", "") or "").strip().lower()
    readonly_param = request.args.get("readonly", type=int) == 1
    readonly = (role not in ("admin", "superadmin")) or readonly_param
    if not readonly and role not in ("admin", "superadmin"):
        flash("Access denied", "danger")
        return redirect(url_for("inventory.wo_list"))

    wo = WorkOrder.query.get_or_404(wo_id)

    technicians = _query_technicians()

    # preselect
    selected_tech_id = None
    selected_tech_username = None
    if getattr(wo, "technician_id", None):
        selected_tech_id = int(wo.technician_id)
        for tid, uname in technicians:
            if tid == selected_tech_id:
                selected_tech_username = uname
                break
    if selected_tech_id is None:
        wo_name = (wo.technician_name or "").strip().lower()
        if wo_name:
            for tid, uname in technicians:
                if (uname or "").strip().lower() == wo_name:
                    selected_tech_id = tid
                    selected_tech_username = uname
                    break

    # units payload (unchanged)
    units = []
    for u in (getattr(wo, "units", []) or []):
        rows = []
        for p in (getattr(u, "parts", []) or []):
            rows.append({
                "id": getattr(p, "id", None),
                "part_number": getattr(p, "part_number", "") or "",
                "part_name": getattr(p, "part_name", "") or "",
                "quantity": int(getattr(p, "quantity", 0) or 0),
                "alt_numbers": getattr(p, "alt_numbers", "") or "",
                "warehouse": getattr(p, "warehouse", "") or "",
                "supplier": getattr(p, "supplier", "") or "",
                "backorder_flag": bool(getattr(p, "backorder_flag", False)),
                "line_status": getattr(p, "line_status", "") or "search_ordered",
                "unit_cost": (
                    float(getattr(p, "unit_cost")) if getattr(p, "unit_cost") is not None else None
                ),
            })
        if not rows:
            rows = [{
                "id": None,
                "part_number": "", "part_name": "", "quantity": 1,
                "alt_numbers": "",
                "warehouse": "", "supplier": "",
                "backorder_flag": False, "line_status": "search_ordered",
                "unit_cost": 0.0,
            }]
        units.append({
            "id": getattr(u, "id", None),
            "brand": getattr(u, "brand", "") or "",
            "model": getattr(u, "model", "") or "",
            "serial": getattr(u, "serial", "") or "",
            "rows": rows,
        })

    if not units:
        units = [{
            "brand":  wo.brand or "",
            "model":  wo.model or "",
            "serial": wo.serial or "",
            "rows": [{
                "id": None,
                "part_number": "", "part_name": "", "quantity": 1,
                "alt_numbers": "",
                "warehouse": "", "supplier": "",
                "backorder_flag": False, "line_status": "search_ordered",
                "unit_cost": 0.0,
            }],
        }]

    recent_suppliers = session.get("recent_suppliers", []) or []

    return render_template(
        "wo_form_units.html",
        wo=wo,
        units=units,
        recent_suppliers=recent_suppliers,
        readonly=readonly,
        technicians=technicians,
        selected_tech_id=selected_tech_id,
        selected_tech_username=selected_tech_username,
    )

@inventory_bp.post("/work_orders/<int:wo_id>/delete", endpoint="wo_delete")
@login_required
def wo_delete(wo_id: int):
    if getattr(current_user, "role", "") != "superadmin":
        flash("Access denied", "danger")
        return redirect(url_for("inventory.wo_detail", wo_id=wo_id))

    def table_exists(name: str) -> bool:
        try:
            row = db.session.execute(
                text("SELECT name FROM sqlite_master WHERE type='table' AND name=:n"),
                {"n": name}
            ).fetchone()
            return bool(row)
        except Exception:
            return False

    try:
        # 1) Пробуем через модели (если у тебя они есть)
        try:
            from models import WorkOrder  # подстрой, если модуль другой
            wo = WorkOrder.query.get_or_404(wo_id)
            db.session.delete(wo)
            db.session.commit()
        except Exception as model_err:
            current_app.logger.debug("Model delete failed, fallback to SQL: %s", model_err)

            # 2) Фоллбэк на SQL — удаляем дочерние записи, но только если таблицы существуют
            for t in ("issued_part_records", "work_order_parts", "issued_batches", "job_index"):
                if table_exists(t):
                    db.session.execute(text(f"DELETE FROM {t} WHERE work_order_id=:id"), {"id": wo_id})

            if table_exists("work_orders"):
                db.session.execute(text("DELETE FROM work_orders WHERE id=:id"), {"id": wo_id})

            db.session.commit()

        flash(f"Work Order #{wo_id} deleted.", "success")
        return redirect(url_for("inventory.wo_list"))

    except Exception as e:
        db.session.rollback()
        current_app.logger.exception("Failed to delete work order %s", wo_id)
        flash("Failed to delete work order.", "danger")
        return redirect(url_for("inventory.wo_detail", wo_id=wo_id))

def _parse_units_form(form):
    """
    Ожидаем схему имён:
      units[U][brand], units[U][model], units[U][serial]
      units[U][rows][I][part_number], ... quantity, alt_numbers, supplier, backorder_flag, line_status
    """
    import re
    units = {}
    for k, v in form.items():
        m = re.match(r"^units\[(\d+)\]\[(brand|model|serial)\]$", k)
        if m:
            ui = int(m.group(1))
            units.setdefault(ui, {"rows": []})
            units[ui][m.group(2)] = v.strip()
            continue

        m = re.match(r"^units\[(\d+)\]\[rows\]\[(\d+)\]\[(\w+)\]$", k)
        if m:
            ui = int(m.group(1)); ri = int(m.group(2)); field = m.group(3)
            while len(units.setdefault(ui, {"rows": []})["rows"]) <= ri:
                units[ui]["rows"].append({})
            units[ui]["rows"][ri][field] = v.strip()
            continue

    # чекбоксы backorder_flag могут не прийти — нормализуем
    for u in units.values():
        for row in u["rows"]:
            row["quantity"] = int(row.get("quantity") or 0)
            row["backorder_flag"] = bool(row.get("backorder_flag") in ("on", "true", "1"))
            row["line_status"] = (row.get("line_status") or "search_ordered")
    # превращаем в упорядоченный список
    out = [units[i] for i in sorted(units.keys())]
    return out
@inventory_bp.get("/api/part_lookup")
@login_required
def api_part_lookup():
    from sqlalchemy import func
    from models import Part
    pn = (request.args.get("pn") or "").strip().upper()
    if not pn:
        return jsonify({"found": False})
    part = Part.query.filter(func.upper(Part.part_number) == pn).first()
    if not part:
        return jsonify({"found": False, "stock_hint": "—"})
    on_hand = int(getattr(part, "on_hand", 0) or getattr(part, "quantity", 0) or 0)
    stock_hint = "STOCK" if on_hand > 0 else "WAIT"
    wh = getattr(part, "location", None) or getattr(part, "wh", None) or ""
    return jsonify({"found": True, "name": getattr(part, "name", "") or "", "wh": wh, "stock_hint": stock_hint})


# @inventory_bp.post("/work_orders/savex")
# @login_required
# def wo_savex():
#     if getattr(current_user, "role", "") not in ("admin", "superadmin"):
#         flash("Access denied", "danger")
#         return redirect(url_for("inventory.wo_list"))
#
#     from models import WorkOrder, WorkUnit, WorkOrderPart, Part
#     from extensions import db
#     from sqlalchemy import func
#     from datetime import date
#
#     f = request.form
#
#     def _f(x, default=0.0):
#         try: return float(x)
#         except: return float(default)
#
#     def _i(x, default=0):
#         try: return int(x)
#         except: return int(default)
#
#     def _clip(s: str, n: int) -> str:
#         return (s or "").strip()[:n]
#
#     wo_id = (f.get("wo_id") or "").strip()
#     tech  = (f.get("technician_name") or "").strip().upper()
#     job_numbers = (f.get("job_numbers") or "").strip()
#     job_type = (f.get("job_type") or "BASE").strip().upper()
#     status_hdr = (f.get("status") or "search_ordered").strip()
#     delivery_fee   = _f(f.get("delivery_fee"), 0)
#     markup_percent = _f(f.get("markup_percent"), 0)
#
#     # Парсим форму
#     try:
#         from inventory.utils import _parse_units_form
#         units_payload = _parse_units_form(f) or []
#     except Exception:
#         units_payload = []
#
#     if not units_payload:
#         flash("Invalid or empty form submission.", "danger")
#         return redirect(url_for("inventory.wo_list"))
#
#     # Проверка supplier
#     missing_pns = []
#     for u in units_payload:
#         for r in (u.get("rows") or []):
#             if (r.get("part_number") or "").strip() and not (r.get("supplier") or "").strip():
#                 missing_pns.append(r["part_number"])
#     if missing_pns:
#         flash(f"Supplier is required for: {', '.join(set(missing_pns))}", "danger")
#         if wo_id:
#             return redirect(url_for("inventory.wo_edit", wo_id=int(wo_id)))
#         else:
#             return redirect(url_for("inventory.wo_new"))
#
#     # Создание/обновление WO
#     if wo_id:
#         wo = WorkOrder.query.get_or_404(int(wo_id))
#     else:
#         wo = WorkOrder()
#         db.session.add(wo)
#
#     wo.technician_name = tech
#     wo.job_numbers = job_numbers
#     wo.job_type = job_type
#     wo.status = status_hdr if status_hdr in ("search_ordered", "ordered", "done") else "search_ordered"
#     wo.delivery_fee = delivery_fee
#     wo.markup_percent = markup_percent
#
#     # Удаляем старые юниты
#     if wo_id:
#         for u in list(getattr(wo, "units", []) or []):
#             db.session.delete(u)
#         db.session.flush()
#
#     # Пересоздание юнитов
#     for u in units_payload:
#         unit = WorkUnit(
#             work_order=wo,
#             brand=(u.get("brand") or "").strip(),
#             model=_clip(u.get("model"), 25),
#             serial=_clip(u.get("serial"), 25),
#         )
#         db.session.add(unit)
#         db.session.flush()
#
#         for r in (u.get("rows") or []):
#             pn = _clip((r.get("part_number") or "").upper(), 80)
#             if not pn:
#                 continue
#             qty = _i(r.get("quantity") or 0)
#             supplier_val = _clip((r.get("supplier") or "").strip(), 80)
#
#             part_rec = Part.query.filter(func.upper(Part.part_number) == pn).first()
#             part_name = _clip(r.get("part_name") or getattr(part_rec, "name", "") or "", 120)
#             warehouse = _clip(r.get("warehouse") or getattr(part_rec, "location", "") or "", 120)
#             unit_cost = _f(r.get("unit_cost") or 0)
#             alt_raw = (r.get("alt_numbers") or "").strip()
#
#             wop = WorkOrderPart(
#                 work_order=wo,
#                 unit=unit,
#                 part_number=pn,
#                 part_name=part_name,
#                 quantity=qty,
#                 supplier=supplier_val,
#                 warehouse=warehouse,
#                 backorder_flag=bool(r.get("backorder_flag")),
#             )
#
#             # 🔹 ORD: флаг и дата
#             ord_on = bool(r.get("ordered_flag")) or ((r.get("line_status") or "").lower() == "ordered")
#             if hasattr(wop, "line_status"):
#                 wop.line_status = "ordered" if ord_on else "search_ordered"
#             if hasattr(wop, "status"):
#                 wop.status = "ordered" if ord_on else "search_ordered"
#             if hasattr(wop, "ordered_date"):
#                 # при установке ORD — ставим сегодняшнюю дату
#                 wop.ordered_date = date.today() if ord_on else None
#
#             if hasattr(wop, "unit_cost"):
#                 wop.unit_cost = unit_cost
#
#             db.session.add(wop)
#
#     db.session.commit()
#     flash("Work Order saved.", "success")
#     return redirect(url_for("inventory.wo_detail", wo_id=wo.id))

@inventory_bp.post("/work_orders/<int:wo_id>/units/<int:unit_id>/issue_instock")
@login_required
def wo_issue_instock_unit(wo_id, unit_id):
    if getattr(current_user, "role", "") not in ("admin", "superadmin"):
        flash("Access denied", "danger")
        return redirect(url_for("inventory.wo_detail", wo_id=wo_id))

    from datetime import datetime, timedelta
    from sqlalchemy import and_
    from flask import request
    from extensions import db
    from models import WorkOrder, Part, IssuedPartRecord

    wo = WorkOrder.query.get_or_404(wo_id)
    unit = next((u for u in (wo.units or []) if u.id == unit_id), None)
    if not unit:
        flash("Unit not found", "danger")
        return redirect(url_for("inventory.wo_detail", wo_id=wo_id))

    # флаг автосмены статуса
    set_status = (request.form.get("set_status") or "").strip().lower()

    rows = compute_availability_unit(unit, wo.status)

    items = []
    for r in rows:
        if int(r.get("issue_now", 0)) > 0:
            pn = (r.get("part_number") or "").strip().upper()
            if not pn:
                continue

            part = Part.query.filter_by(part_number=pn).first()
            if not part:
                continue

            # ✅ фиксируем себестоимость склада в момент выдачи
            try:
                real_cost = float(part.unit_cost or 0.0)
            except Exception:
                real_cost = 0.0

            items.append({
                "part_id": part.id,
                "qty": int(r["issue_now"]),
                "unit_price": real_cost,   # пробрасываем стоимость склада
            })

    if not items:
        flash("Nothing available to issue for this unit.", "warning")
        return redirect(url_for("inventory.wo_detail", wo_id=wo_id))

    # --- создаём строки выдачи ---
    issue_date, maybe_records = _issue_records_bulk(
        issued_to=wo.technician_name,
        reference_job=wo.canonical_job,
        items=items
    )

    # поддержка обоих вариантов возврата (_issue_records_bulk может вернуть count)
    if isinstance(maybe_records, (list, tuple)) and maybe_records and hasattr(maybe_records[0], "id"):
        new_records = list(maybe_records)
    else:
        # fallback: собрать только что созданные строки по окну +/-2 минуты
        t0 = issue_date - timedelta(minutes=2)
        t1 = issue_date + timedelta(minutes=2)
        new_records = IssuedPartRecord.query.filter(
            and_(
                IssuedPartRecord.issued_to == wo.technician_name,
                IssuedPartRecord.reference_job == wo.canonical_job,
                IssuedPartRecord.invoice_number.is_(None),
                IssuedPartRecord.issue_date >= t0,
                IssuedPartRecord.issue_date <= t1,
            )
        ).all()

    if not new_records:
        # даже если не нашли "new_records" (крайне редко), мы всё равно можем завершить
        if set_status == "done":
            try:
                wo.status = "done"
                db.session.commit()
            except Exception:
                db.session.rollback()
        flash("Issued items saved, but could not collect records for invoice.", "warning")
        return redirect(url_for("inventory.wo_detail", wo_id=wo_id))

    # --- формируем инвойс/батч ---
    try:
        batch = _ensure_invoice_number_for_records(
            records=new_records,
            issued_to=wo.technician_name,
            issued_by=getattr(current_user, "username", "system"),
            reference_job=wo.canonical_job,
            issue_date=datetime.utcnow(),
            location=None
        )
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        if set_status == "done":
            try:
                wo.status = "done"
                db.session.commit()
            except Exception:
                db.session.rollback()
        flash(f"Error creating invoice batch: {e}", "danger")
        return redirect(url_for("inventory.wo_detail", wo_id=wo_id))

    # --- обновляем статус WO ---
    if set_status == "done":
        try:
            wo.status = "done"
            db.session.commit()
        except Exception:
            db.session.rollback()
    else:
        try:
            avail_all = compute_availability_multi(wo)
            still_wait = any(
                int(x.get("on_hand", 0)) < int(x.get("requested", 0))
                for x in avail_all
            )
            if not still_wait:
                wo.status = "done"
                db.session.commit()
        except Exception:
            db.session.rollback()

    # --- редирект прямо на отчёт по инвойсу ---
    return redirect(f"/reports/{batch.invoice_number}", code=303)

@inventory_bp.post("/work_orders/<int:wo_id>/status")
@login_required
def wo_set_status(wo_id):
    if getattr(current_user, "role", "") not in ("admin", "superadmin"):
        flash("Access denied", "danger")
        return redirect(url_for("inventory.wo_detail", wo_id=wo_id))
    new_status = (request.form.get("status") or "").strip()
    if new_status not in ("search_ordered","ordered","done"):
        flash("Invalid status", "warning")
        return redirect(url_for("inventory.wo_detail", wo_id=wo_id))
    wo = WorkOrder.query.get_or_404(wo_id)
    wo.status = new_status
    db.session.commit()
    flash(f"Status set to {new_status}.", "success")
    return redirect(url_for("inventory.wo_detail", wo_id=wo_id))



@inventory_bp.post("/issue/batch")
@login_required
def issue_batch():
    # 🔐 только admin/superadmin
    if getattr(current_user, "role", "") not in ("admin", "superadmin"):
        return jsonify({"ok": False, "error": "Access denied"}), 403
    try:
        payload = request.get_json(force=True, silent=False) or {}
        issued_to = (payload.get("issued_to") or "").strip()
        reference_job = (payload.get("reference_job") or "").strip()
        items = payload.get("items") or []

        if not issued_to:
            return jsonify({"ok": False, "error": "issued_to is required"}), 400
        if not reference_job:
            return jsonify({"ok": False, "error": "reference_job is required"}), 400

        # 🧾 логируем состав выдачи (без персональных данных)
        try:
            safe_items = [
                {"part_id": int(i.get("part_id", 0)), "qty": int(i.get("qty", 0)),
                 "unit_price": None if i.get("unit_price") in (None, "") else float(i.get("unit_price"))}
                for i in items
            ]
            current_app.logger.info("ISSUE_BATCH by=%s job=%s items=%s",
                                    getattr(current_user, "username", "?"),
                                    reference_job, safe_items)
        except Exception:
            pass

        issue_date, created = _issue_records_bulk(issued_to, reference_job, items)
        today = issue_date.date().isoformat()
        params = urlencode({
            "start_date": today, "end_date": today,
            "recipient": issued_to, "reference_job": reference_job
        })
        return jsonify({
            "ok": True,
            "created": created,   # ← для тоста
            "redirect": f"/reports_grouped?{params}"
        }), 200

    except ValueError as ve:
        db.session.rollback()
        return jsonify({"ok": False, "error": str(ve)}), 400
    except SQLAlchemyError:
        db.session.rollback()
        return jsonify({"ok": False, "error": "DB error"}), 500
    except Exception as e:
        db.session.rollback()
        return jsonify({"ok": False, "error": str(e)}), 500


@inventory_bp.post("/issue/line/<int:part_id>")
@login_required
def issue_line(part_id):
    """
    Выдать одну позицию (когда нужно выдать конкретный Part сейчас).
    Принимает JSON или form-data с полями:
      - qty (int) — обязательное, > 0
      - issued_to (str) — обязательное
      - reference_job (str) — обязательное
      - unit_price (float) — опционально (если уже учтены доставка/наценка)
    Возвращает:
      - JSON {"ok": True, "created": 1, "redirect": "..."} для JSON-запросов
      - redirect на /reports_grouped для форм
    """
    # 🔐 Разрешаем только admin / superadmin
    if getattr(current_user, "role", "") not in ("admin", "superadmin"):
        # Для JSON-запросов — JSON, для форм — flash + редирект
        if request.is_json:
            return jsonify({"ok": False, "error": "Access denied"}), 403
        flash("Access denied", "danger")
        return redirect(url_for("inventory.dashboard"))

    try:
        # Универсально читаем данные (JSON или форма)
        data = (request.get_json(silent=True) or {}) if request.is_json else (request.form or {})
        qty = int(data.get("qty") or 0)
        issued_to = (data.get("issued_to") or "").strip()
        reference_job = (data.get("reference_job") or "").strip()

        # unit_price — опционально
        unit_price_raw = data.get("unit_price")
        unit_price = None
        if unit_price_raw not in (None, ""):
            unit_price = float(unit_price_raw)

        # Валидация
        if qty <= 0:
            raise ValueError("qty must be > 0")
        if not issued_to or not reference_job:
            raise ValueError("issued_to and reference_job are required")

        # 🧾 Лог выдачи одной строки (без чувствительных данных)
        try:
            current_app.logger.info(
                "ISSUE_LINE by=%s job=%s part_id=%s qty=%s price=%s",
                getattr(current_user, "username", "?"),
                reference_job, part_id, qty, unit_price
            )
        except Exception:
            pass

        # Выдаём ровно одну позицию
        items = [{"part_id": int(part_id), "qty": qty, "unit_price": unit_price}]
        issue_date, _created = _issue_records_bulk(issued_to, reference_job, items)  # min(qty, on_hand) внутри

        # Готовим редирект на сгруппированный отчёт за сегодня
        today = issue_date.date().isoformat()
        params = urlencode({
            "start_date": today,
            "end_date": today,
            "recipient": issued_to,
            "reference_job": reference_job
        })
        target = f"/reports_grouped?{params}"

        if request.is_json:
            return jsonify({"ok": True, "created": 1, "redirect": target}), 200
        else:
            return redirect(target, code=303)

    except ValueError as ve:
        db.session.rollback()
        if request.is_json:
            return jsonify({"ok": False, "error": str(ve)}), 400
        flash(f"Issue failed: {ve}", "danger")
        return redirect(url_for("inventory.dashboard"))

    except Exception as e:
        db.session.rollback()
        if request.is_json:
            return jsonify({"ok": False, "error": "Internal error"}), 500
        current_app.logger.exception("ISSUE_LINE failed: %s", e)
        flash("Issue failed: internal error", "danger")
        return redirect(url_for("inventory.dashboard"))


@inventory_bp.get("/issue_ui")
@login_required
def issue_ui():
    parts = Part.query.order_by(Part.part_number).limit(200).all()  # для примера
    technician_name = getattr(current_user, "username", "TECH")     # замени как надо
    canonical_ref = "TESTJOB123"                                    # подставишь свой JOB
    return render_template("issue_ui.html", parts=parts,
                           technician_name=technician_name,
                           canonical_ref=canonical_ref)


# ----------------- Dashboard -----------------

@inventory_bp.route('/api/part/<part_number>', methods=['GET'])
@login_required
def get_part_by_number(part_number):
    """
    Вернёт информацию о детали по Part #.
    Ответ (200):
      {
        "id": int,
        "name": str,
        "location": str | null,
        "unit_cost": float,
        "quantity": int
      }
    Или (404): {"error": "Not found"}
    """
    try:
        pn = (part_number or "").strip().upper()
        if not pn:
            return jsonify({"error": "Part number is required"}), 400

        part = Part.query.filter_by(part_number=pn).first()
        if not part:
            return jsonify({"error": "Not found"}), 404

        # Нормализуем типы в ответе
        return jsonify({
            "id": int(part.id),
            "name": part.name or "",
            "location": part.location,
            "unit_cost": float(part.unit_cost or 0.0),
            "quantity": int(part.quantity or 0),
        }), 200

    except Exception as e:
        current_app.logger.exception("api/part failed for %r: %s", part_number, e)
        return jsonify({"error": "Internal error"}), 500




@inventory_bp.route('/api/part_lookup')
@login_required
def part_lookup():
    part_number = request.args.get('part_number', '').strip().upper()
    part = Part.query.filter_by(part_number=part_number).first()
    if part:
        return {
            'found': True,
            'id': part.id,
            'name': part.name,
            'quantity': part.quantity
        }
    return { 'found': False }

@inventory_bp.route('/')
@login_required
def dashboard():
    search_query = request.args.get('search', '').strip()
    parts = Part.query

    if search_query:
        parts = parts.filter(
            or_(
                Part.part_number.ilike(f"%{search_query}%"),
                Part.name.ilike(f"%{search_query}%")
            )
        )

    parts = parts.all()
    return render_template('index.html', parts=parts, search_query=search_query)

# @inventory_bp.route('/inventory_summary')
# @login_required
# def inventory_summary():
#     parts = Part.query.filter(Part.quantity > 0).all()
#
#     locations = defaultdict(lambda: {
#         'total_quantity': 0,
#         'total_value': 0.0,
#     })
#
#     for part in parts:
#         loc = part.location or 'Unknown'
#         locations[loc]['total_quantity'] += part.quantity
#         locations[loc]['total_value'] += part.quantity * part.unit_cost
#
#     grand_total_quantity = sum(data['total_quantity'] for data in locations.values())
#     grand_total_value = sum(data['total_value'] for data in locations.values())
#
#     return render_template('inventory_summary.html',
#                            locations=locations,
#                            grand_total_quantity=grand_total_quantity,
#                            grand_total_value=grand_total_value)

@inventory_bp.route('/dashboard/location_report')
@login_required
def location_report():
    parts = Part.query.filter(Part.quantity > 0).all()

    locations = defaultdict(lambda: {
        'parts': [],
        'total_quantity': 0,
        'total_value': 0.0,
    })

    for part in parts:
        loc = part.location or 'Unknown'
        locations[loc]['parts'].append(part)
        locations[loc]['total_quantity'] += part.quantity
        locations[loc]['total_value'] += part.quantity * part.unit_cost

    grand_total_quantity = sum(data['total_quantity'] for data in locations.values())
    grand_total_value = sum(data['total_value'] for data in locations.values())

    return render_template('location_report.html',
                           locations=locations,
                           grand_total_quantity=grand_total_quantity,
                           grand_total_value=grand_total_value)

# --- Печать детального отчёта по локациям (весь список товаров) ---
@inventory_bp.route('/dashboard/location_report/print')
@login_required
def print_location_report():
    parts = Part.query.filter(Part.quantity > 0).all()

    locations = defaultdict(lambda: {
        'parts': [],
        'total_quantity': 0,
        'total_value': 0.0,
    })

    for part in parts:
        loc = part.location or 'Unknown'
        locations[loc]['parts'].append(part)
        locations[loc]['total_quantity'] += part.quantity
        locations[loc]['total_value'] += part.quantity * part.unit_cost

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(letter))
    elements = []

    styles = getSampleStyleSheet()
    title_style = styles['Heading2']

    for loc, data in locations.items():
        elements.append(Paragraph(f"Location: {loc}", title_style))
        elements.append(Paragraph(f"Total Quantity: {data['total_quantity']}", styles['Normal']))
        elements.append(Paragraph(f"Total Value: ${data['total_value']:.2f}", styles['Normal']))
        elements.append(Spacer(1, 12))

        table_data = [["Part Number", "Name", "Quantity", "Unit Cost", "Total Cost"]]
        for part in data['parts']:
            table_data.append([
                part.part_number,
                part.name,
                str(part.quantity),
                f"${part.unit_cost:.2f}",
                f"${part.quantity * part.unit_cost:.2f}"
            ])
        # Итог по локации
        table_data.append([
            "Location Total", "",
            str(data['total_quantity']), "", f"${data['total_value']:.2f}"
        ])

        table = Table(table_data, hAlign='LEFT')
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (2, 1), (-1, -1), 'RIGHT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, -1), (-1, -1), colors.lightblue),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 0.25, colors.black),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 20))

    doc.build(elements)
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name="location_report.pdf", mimetype='application/pdf')

# --- Печать сводного отчёта (итог по локациям без деталей) ---
@inventory_bp.route('/dashboard/inventory_summary/print')
@login_required
def print_inventory_summary():
    parts = Part.query.filter(Part.quantity > 0).all()

    locations = defaultdict(lambda: {
        'total_quantity': 0,
        'total_value': 0.0,
    })

    for part in parts:
        loc = part.location or 'Unknown'
        locations[loc]['total_quantity'] += part.quantity
        locations[loc]['total_value'] += part.quantity * part.unit_cost

    grand_total_quantity = sum(data['total_quantity'] for data in locations.values())
    grand_total_value = sum(data['total_value'] for data in locations.values())

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []

    styles = getSampleStyleSheet()
    elements.append(Paragraph("Inventory Summary Report by Location", styles['Heading2']))
    elements.append(Spacer(1, 12))

    table_data = [["Location", "Total Quantity", "Total Value"]]
    for loc, data in locations.items():
        table_data.append([
            loc,
            str(data['total_quantity']),
            f"${data['total_value']:.2f}"
        ])
    # Итог
    table_data.append([
        "Grand Total",
        str(grand_total_quantity),
        f"${grand_total_value:.2f}"
    ])

    table = Table(table_data, hAlign='LEFT')
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, -1), (-1, -1), colors.lightgrey),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.black),
    ]))

    elements.append(table)
    doc.build(elements)
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name="inventory_summary.pdf", mimetype='application/pdf')



@inventory_bp.route('/add', methods=['GET', 'POST'])
@login_required
def add_part():
    if request.method == 'POST':
        part_number = request.form['part_number'].strip().upper()
        name = request.form['name'].strip().upper()
        quantity = int(request.form['quantity'])
        unit_cost = float(request.form['unit_cost'])
        location = request.form['location'].strip().upper()

        existing = Part.query.filter_by(part_number=part_number).first()
        if existing:
            existing.quantity += quantity
            existing.unit_cost = unit_cost
            existing.name = name
            existing.location = location
        else:
            new_part = Part(
                name=name,
                part_number=part_number,
                quantity=quantity,
                unit_cost=unit_cost,
                location=location
            )
            db.session.add(new_part)

        db.session.commit()
        flash('Part saved successfully.', 'success')
        return redirect(url_for('inventory.dashboard'))

    return render_template('add_part.html')


# ----------------- Issue Part -----------------
@inventory_bp.route('/issue', methods=['GET', 'POST'])
@login_required
def issue_part():
    parts = Part.query.filter(Part.quantity > 0).all()

    if request.method == 'POST':
        import json
        try:
            all_parts = json.loads(request.form.get('all_parts_json', '[]'))
        except json.JSONDecodeError:
            flash('Invalid part data.', 'danger')
            return redirect(url_for('.issue_part'))

        if not all_parts:
            flash('No parts to issue.', 'warning')
            return redirect(url_for('.issue_part'))

        for item in all_parts:
            part = Part.query.get(item['part_id'])
            if not part or part.quantity < item['quantity']:
                flash(f"Not enough stock for {item.get('part_number', 'UNKNOWN')}", 'danger')
                return redirect(url_for('.issue_part'))

            part.quantity -= item['quantity']

            record = IssuedPartRecord(
                part_id=part.id,
                quantity=item['quantity'],
                issued_to=item['recipient'],
                reference_job=item['reference_job'],
                issued_by=current_user.username,
                issue_date=datetime.utcnow(),
                unit_cost_at_issue=part.unit_cost  # фиксируем цену на момент выдачи
            )
            db.session.add(record)

        db.session.commit()
        flash('All parts issued successfully.', 'success')

        # >>> ЖЁСТКИЙ РЕДИРЕКТ СРАЗУ В ОТЧЁТ (без url_for, чтобы исключить любые конфликты)
        today = datetime.utcnow().date().isoformat()
        first = all_parts[0]
        recipient = (first.get('recipient') or '').strip()
        reference_job = (first.get('reference_job') or '').strip()

        params = {'start_date': today, 'end_date': today}
        if recipient:
            params['recipient'] = recipient
        if reference_job:
            params['reference_job'] = reference_job

        # Итог: /reports_grouped?start_date=YYYY-MM-DD&end_date=YYYY-MM-DD&recipient=...&reference_job=...
        return redirect('/reports_grouped?' + urlencode(params), code=303)

    return render_template('issue_part.html', parts=parts)



# ----------------- Reports -----------------



@inventory_bp.route('/reports', methods=['GET', 'POST'])
@login_required
def reports():
    query = IssuedPartRecord.query.join(Part)
    start_date = request.form.get('start_date')
    end_date = request.form.get('end_date')
    recipient = request.form.get('recipient')
    reference_job = request.form.get('reference_job')

    if start_date:
        query = query.filter(IssuedPartRecord.issue_date >= start_date)
    if end_date:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
        query = query.filter(IssuedPartRecord.issue_date <= end_dt)
    if recipient:
        query = query.filter(IssuedPartRecord.issued_to.ilike(f'%{recipient}%'))
    if reference_job:
        query = query.filter(IssuedPartRecord.reference_job.ilike(f'%{reference_job}%'))

    records = query.order_by(IssuedPartRecord.issue_date.desc()).all()

    # Группируем записи по ключу
    invoices_map = defaultdict(lambda: {
        'issued_to': '',
        'reference_job': '',
        'issued_by': '',
        'issue_date': None,
        'items': [],
        'total_value': 0.0,
    })

    grand_total = 0.0

    for r in records:
        key = (r.issued_to, r.reference_job or '', r.issued_by, r.issue_date.date())
        inv = invoices_map[key]
        inv['issued_to'] = r.issued_to
        inv['reference_job'] = r.reference_job
        inv['issued_by'] = r.issued_by
        inv['issue_date'] = r.issue_date
        inv['items'].append(r)
        line_total = r.quantity * r.unit_cost_at_issue
        inv['total_value'] += line_total
        grand_total += line_total

    invoices = list(invoices_map.values())

    return render_template(
        'reports_grouped.html',
        invoices=invoices,
        total=grand_total,
        start_date=start_date,
        end_date=end_date,
        recipient=recipient,
        reference_job=reference_job
    )
# inventory/routes.py
@inventory_bp.route('/reports_grouped', methods=['GET', 'POST'])
@login_required
def reports_grouped():
    from collections import defaultdict
    from datetime import datetime, time
    from sqlalchemy.orm import selectinload
    from sqlalchemy import func, or_
    from flask import render_template, request
    from flask_login import current_user
    from extensions import db
    from models import IssuedPartRecord, IssuedBatch, Part

    def _parse_date_ymd(s: str | None):
        if not s:
            return None
        try:
            return datetime.strptime(s, "%Y-%m-%d")
        except Exception:
            return None

    # ---------- Параметры (GET/POST) ----------
    params         = request.values
    start_date_s   = (params.get('start_date') or '').strip()
    end_date_s     = (params.get('end_date') or '').strip()
    recipient_raw  = (params.get('recipient') or '').strip() or None
    reference_job  = (params.get('reference_job') or '').strip() or None
    invoice_s      = (params.get('invoice_number') or params.get('invoice') or params.get('invoice_no') or '').strip()
    location       = (params.get('location') or '').strip() or None

    # роль/текущий пользователь
    role_low = (getattr(current_user, "role", "") or "").strip().lower()
    me_user  = (getattr(current_user, "username", "") or "").strip()

    # ТЕХНИК: принудительно фильтруем только по себе
    recipient_effective = me_user if role_low == "technician" else recipient_raw

    # даты → границы дня
    start_dt_raw = _parse_date_ymd(start_date_s)
    end_dt_raw   = _parse_date_ymd(end_date_s)
    start_dt = datetime.combine(start_dt_raw.date(), time.min) if start_dt_raw else None
    end_dt   = datetime.combine(end_dt_raw.date(),   time.max) if end_dt_raw   else None

    # invoice_number (опционально)
    try:
        invoice_no = int(invoice_s) if invoice_s else None
    except ValueError:
        invoice_no = None

    # ---------- Запрос с подзагрузкой связей ----------
    q = (
        db.session.query(IssuedPartRecord)
        .join(Part, IssuedPartRecord.part_id == Part.id)
        .outerjoin(IssuedBatch, IssuedBatch.id == IssuedPartRecord.batch_id)
        .options(
            selectinload(IssuedPartRecord.part),
            selectinload(IssuedPartRecord.batch),
        )
    )

    # Фильтр по "кому выдано"
    if recipient_effective:
        if role_low == "technician":
            q = q.filter(
                or_(
                    func.trim(IssuedPartRecord.issued_to) == recipient_effective,
                    func.trim(IssuedBatch.issued_to)     == recipient_effective,
                )
            )
        else:
            like = f"%{recipient_effective}%"
            q = q.filter(
                or_(
                    IssuedPartRecord.issued_to.ilike(like),
                    IssuedBatch.issued_to.ilike(like),
                )
            )

    if reference_job:
        q = q.filter(IssuedPartRecord.reference_job.ilike(f'%{reference_job}%'))
    if start_dt:
        q = q.filter(IssuedPartRecord.issue_date >= start_dt)
    if end_dt:
        q = q.filter(IssuedPartRecord.issue_date <= end_dt)
    if invoice_no is not None:
        q = q.filter(IssuedPartRecord.invoice_number == invoice_no)
    if location:
        q = q.filter(IssuedPartRecord.location == location)

    rows = q.order_by(
        IssuedPartRecord.issue_date.desc(),
        IssuedPartRecord.id.desc()
    ).all()

    # ---------- Группировка: batch/legacy ----------
    grouped = defaultdict(list)
    for r in rows:
        if getattr(r, 'batch_id', None):
            # Каждая партия (batch) — отдельная группа
            key = ('BATCH', r.batch_id)
        else:
            # В legacy-ветке добавляем invoice_number в ключ,
            # чтобы разные накладные не слипались.
            inv_num = getattr(r, 'invoice_number', None)
            key = (
                'LEGACY',
                r.issued_to,
                r.reference_job,
                r.issued_by,
                r.issue_date.date(),
                inv_num,  # важно!
            )
        grouped[key].append(r)

    invoices = []
    grand_total = 0.0

    def _is_return_records(items: list[IssuedPartRecord]) -> bool:
        if not items:
            return False
        if any((it.quantity or 0) < 0 for it in items):
            return True
        ref = (items[0].reference_job or '').strip().upper()
        return ref.startswith('RETURN')

    for gkey, items in grouped.items():
        items_sorted = sorted(items, key=lambda it: it.id)
        total_value = sum((it.quantity or 0) * (it.unit_cost_at_issue or 0.0) for it in items_sorted)
        grand_total += total_value

        if gkey[0] == 'BATCH':
            batch = items_sorted[0].batch
            # На всякий случай: если у батча номера нет, возьмём из строки
            batch_inv_number = batch.invoice_number or next(
                (it.invoice_number for it in reversed(items_sorted) if (it.invoice_number or 0) > 0),
                None
            )
            inv = {
                'id': f'B{batch.id}',
                'issued_to': batch.issued_to,
                'reference_job': batch.reference_job,
                'issued_by': batch.issued_by,
                'issue_date': batch.issue_date,
                'invoice_number': batch_inv_number,
                'location': batch.location,
                'items': items_sorted,
                'total_value': total_value,
                'is_return': _is_return_records(items_sorted),
                '_sort_dt': max((it.issue_date for it in items_sorted if it.issue_date), default=batch.issue_date),
                '_sort_id': max((it.id for it in items_sorted), default=0),
            }
        else:
            # gkey = ('LEGACY', issued_to, ref_job, issued_by, day, inv_num)
            _, issued_to, ref_job, issued_by, day, inv_num = gkey
            issue_dt = datetime.combine(day, time.min)
            first = items_sorted[0]

            # ---- Fallback для номера инвойса в legacy-группах ----
            inv_num_from_items = next(
                (it.invoice_number for it in reversed(items_sorted) if (it.invoice_number or 0) > 0),
                None
            )
            resolved_inv_num = inv_num or inv_num_from_items

            if not resolved_inv_num:
                batch_guess = (
                    db.session.query(IssuedBatch)
                    .filter(
                        func.trim(IssuedBatch.issued_to) == (issued_to or ''),
                        func.trim(IssuedBatch.reference_job) == (ref_job or ''),
                        func.trim(IssuedBatch.issued_by) == (issued_by or ''),
                        func.date(IssuedBatch.issue_date) == day
                    )
                    .order_by(IssuedBatch.id.desc())
                    .first()
                )
                if batch_guess and (batch_guess.invoice_number or 0) > 0:
                    resolved_inv_num = batch_guess.invoice_number

            inv = {
                'id': f'K{issued_to}|{issued_by}|{ref_job or ""}|{day.isoformat()}|{resolved_inv_num or ""}',
                'issued_to': issued_to,
                'reference_job': ref_job,
                'issued_by': issued_by,
                'issue_date': issue_dt,
                'invoice_number': resolved_inv_num,
                'location': first.location,
                'items': items_sorted,
                'total_value': total_value,
                'is_return': _is_return_records(items_sorted),
                '_sort_dt': max((it.issue_date for it in items_sorted if it.issue_date), default=issue_dt),
                '_sort_id': max((it.id for it in items_sorted), default=0),
            }

        invoices.append(inv)

    # ---------- Сортировка карточек ----------
    invoices.sort(
        key=lambda g: (
            (g.get('_sort_dt') or datetime.min),
            (g.get('invoice_number') or 0),
            (g.get('_sort_id') or 0),
        ),
        reverse=True
    )

    return render_template(
        'reports_grouped.html',
        invoices=invoices,
        total=grand_total,
        # возвращаем значения в форму
        start_date=start_date_s,
        end_date=end_date_s,
        recipient=(recipient_effective or ''),
        reference_job=reference_job or '',
        invoice=invoice_s or '',
        location=location or '',
    )

@inventory_bp.route("/invoice/pdf")
@login_required
def view_invoice_pdf():
    """
    Печать инвойса.
    Если у группы ещё НЕТ invoice_number и пришли "legacy"-ключи,
    перед печатью аккуратно присваиваем новый номер (через batch),
    коммитим и печатаем уже с этим номером.
    """
    from extensions import db
    from models import IssuedPartRecord, IssuedBatch
    from datetime import datetime, time as _time
    from sqlalchemy import func, or_
    from flask import request, make_response, flash, redirect, url_for

    # ---- вспомогалки ----
    def _next_invoice_number():
        mb = db.session.query(func.coalesce(func.max(IssuedBatch.invoice_number), 0)).scalar() or 0
        ml = db.session.query(func.coalesce(func.max(IssuedPartRecord.invoice_number), 0)).scalar() or 0
        return max(int(mb), int(ml)) + 1

    def _ensure_invoice_number_for_records(records, issued_to, issued_by, reference_job, issue_date, location):
        # уже есть — выходим
        if any(getattr(r, "invoice_number", None) for r in records):
            return getattr(records[0], "invoice_number", None)
        # пробуем твой хелпер (если есть)
        try:
            batch = _create_batch_for_records(
                records=records,
                issued_to=issued_to,
                issued_by=issued_by,
                reference_job=reference_job,
                issue_date=issue_date,
                location=location,
            )
            return getattr(batch, "invoice_number", None)
        except Exception:
            db.session.rollback()

        # fallback — резервируем номер вручную
        for _ in range(5):
            inv_no = _next_invoice_number()
            try:
                with db.session.begin_nested():
                    batch = IssuedBatch(
                        invoice_number=inv_no,
                        issued_to=issued_to,
                        issued_by=issued_by or "system",
                        reference_job=reference_job,
                        issue_date=issue_date,
                        location=(location or None),
                    )
                    db.session.add(batch)
                    db.session.flush()
                    for r in records:
                        r.batch_id = batch.id
                        r.invoice_number = inv_no
                    db.session.flush()
                return inv_no
            except Exception:
                db.session.rollback()
                continue
        raise RuntimeError("Failed to reserve invoice number")

    def _parse_dt_flex(s: str | None):
        if not s:
            return None
        for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y"):
            try:
                return datetime.strptime(s, fmt)
            except Exception:
                pass
        return None

    # ---- входные параметры ----
    inv_s        = (request.args.get("invoice_number") or "").strip()
    issued_to    = (request.args.get("issued_to") or "").strip()
    reference_job= (request.args.get("reference_job") or "").strip() or None
    issued_by    = (request.args.get("issued_by") or "").strip()
    issue_date_s = (request.args.get("issue_date") or "").strip()

    inv_no = int(inv_s) if inv_s.isdigit() else None

    # ---- загрузка строк группы ----
    recs = []
    hdr  = None

    if inv_no is not None:
        # 1) найдём батчи с таким номером
        batch_ids = [bid for (bid,) in db.session.query(IssuedBatch.id)
                     .filter(IssuedBatch.invoice_number == inv_no).all()]

        # 2) строки с этим номером + строки из найденных батчей
        q = IssuedPartRecord.query
        if batch_ids:
            recs = (q.filter(or_(IssuedPartRecord.invoice_number == inv_no,
                                 IssuedPartRecord.batch_id.in_(batch_ids)))
                    .order_by(IssuedPartRecord.id.asc()).all())
        else:
            recs = q.filter_by(invoice_number=inv_no).order_by(IssuedPartRecord.id.asc()).all()

        if not recs:
            flash(f"Invoice #{inv_no} not found.", "warning")
            return redirect(url_for('inventory.reports_grouped'))

        # 3) заголовок — сначала из батча (если есть), иначе из первой строки
        if batch_ids:
            b = db.session.get(IssuedBatch, batch_ids[0])
            hdr = {
                "issued_to": b.issued_to,
                "reference_job": b.reference_job,
                "issued_by": b.issued_by,
                "issue_date": b.issue_date,
                "invoice_number": inv_no,
                "location": getattr(b, "location", None),
            }
        else:
            first = recs[0]
            hdr = {
                "issued_to": first.issued_to,
                "reference_job": first.reference_job,
                "issued_by": first.issued_by,
                "issue_date": first.issue_date,
                "invoice_number": inv_no,
                "location": first.location,
            }

        # 4) ДОП. СЛИЯНИЕ legacy-строк без номера и без batch_id в те же сутки/ключи
        day = hdr["issue_date"].date() if hdr.get("issue_date") else None
        if day:
            extra = (IssuedPartRecord.query
                     .filter(
                         func.trim(IssuedPartRecord.issued_to) == (hdr["issued_to"] or ""),
                         func.trim(IssuedPartRecord.issued_by) == (hdr["issued_by"] or ""),
                         func.trim(IssuedPartRecord.reference_job) == (hdr["reference_job"] or ""),
                         func.date(IssuedPartRecord.issue_date) == day,
                         or_(IssuedPartRecord.invoice_number.is_(None),
                             IssuedPartRecord.invoice_number == 0),
                         IssuedPartRecord.batch_id.is_(None),
                     )
                     .order_by(IssuedPartRecord.id.asc())
                     .all())
            if extra:
                have_ids = {r.id for r in recs}
                recs.extend([r for r in extra if r.id not in have_ids])

        # финальная сортировка
        recs.sort(key=lambda r: r.id)

    else:
        # legacy-поиск по ключам за сутки
        if issued_to and issued_by and issue_date_s:
            dt = _parse_dt_flex(issue_date_s) or datetime.utcnow()
            start = datetime.combine(dt.date(), _time.min)
            end   = datetime.combine(dt.date(), _time.max)
            recs = (IssuedPartRecord.query
                    .filter(IssuedPartRecord.issued_to == issued_to,
                            IssuedPartRecord.issued_by == issued_by,
                            IssuedPartRecord.reference_job == reference_job,
                            IssuedPartRecord.issue_date.between(start, end))
                    .order_by(IssuedPartRecord.id.asc()).all())

    if not recs:
        flash("Invoice lines not found.", "warning")
        return redirect(url_for('inventory.reports_grouped'))

    # если у группы ещё нет номера — назначим перед печатью (как было)
    if inv_no is None and all(getattr(r, "invoice_number", None) is None for r in recs):
        base = recs[0]
        try:
            new_no = _ensure_invoice_number_for_records(
                records=recs,
                issued_to=getattr(base, "issued_to", issued_to),
                issued_by=getattr(base, "issued_by", issued_by),
                reference_job=getattr(base, "reference_job", reference_job),
                issue_date=getattr(base, "issue_date", _parse_dt_flex(issue_date_s) or datetime.utcnow()),
                location=getattr(base, "location", None),
            )
            db.session.commit()
            inv_no = new_no
        except Exception:
            db.session.rollback()
            inv_no = None  # печатаем как есть

    # ---- генерим PDF ----
    pdf_bytes = generate_invoice_pdf(recs, invoice_number=inv_no)
    resp = make_response(pdf_bytes)
    fname = f"INVOICE_{(inv_no or getattr(recs[0],'id', 'NO_NUM')):06d}.pdf" if inv_no is not None else "INVOICE.pdf"
    resp.headers["Content-Type"] = "application/pdf"
    resp.headers["Content-Disposition"] = f'inline; filename="{fname}"'
    return resp

@inventory_bp.route('/reports/update_record/<int:record_id>', methods=['POST'])
@login_required
def update_report_record(record_id):
    if current_user.role not in ['superadmin', 'admin']:
        flash("Access denied", "danger")
        return redirect(url_for('inventory.reports'))

    r = IssuedPartRecord.query.get_or_404(record_id)

    # обновляем разрешённые поля
    if 'issued_to' in request.form:
        v = (request.form.get('issued_to') or '').strip()
        if v: r.issued_to = v

    if 'reference_job' in request.form:
        v = (request.form.get('reference_job') or '').strip() or None
        r.reference_job = v

    if 'unit_cost' in request.form:
        raw = (request.form.get('unit_cost') or '').strip()
        try:
            r.unit_cost_at_issue = float(raw)
        except Exception:
            pass

    if current_user.role == 'superadmin' and 'issue_date' in request.form:
        s = (request.form.get('issue_date') or '').strip()
        if s:
            try:
                # сохраняем ТУ ЖЕ дату с полночью, время не трогаем, чтобы не прыгал порядок
                from datetime import datetime
                d = datetime.strptime(s, '%Y-%m-%d').date()
                r.issue_date = datetime.combine(d, r.issue_date.time())
            except Exception:
                pass

    db.session.commit()

    # КЛЮЧЕВОЕ: после изменения Reference Job/Issued To
    # нужно перерисовать страницы, чтобы кнопки карточки получили НОВЫЕ record_ids[]
    flash("Line saved.", "success")
    return redirect(url_for('inventory.reports'))

# Отмена отдельной позиции
@inventory_bp.route('/reports/cancel/<int:record_id>', methods=['POST'])
@login_required
def cancel_issued_record(record_id):
    if current_user.role not in ['superadmin', 'admin']:
        flash("Access denied", "danger")
        return redirect(url_for('inventory.reports'))

    record = IssuedPartRecord.query.get_or_404(record_id)
    part = Part.query.get(record.part_id)
    if part:
        part.quantity += record.quantity

    db.session.delete(record)
    db.session.commit()

    flash(f"Issued record #{record.id} canceled and stock restored.", "success")
    return redirect(url_for('inventory.reports'))

@inventory_bp.route('/reports/update_invoice', methods=['POST'])
@login_required
def update_invoice():
    """
    Save / Return / Delete from the grouped report.

    Additions in this version:
    - `apply_scope`: 'all' (default) or 'selected'.
    - Invoice number is assigned only for scope='all' and only if ALL rows in the group have no number yet.
    - When scope='all' and there's no number, records are resolved by the group's legacy keys.
    - **Hard cap for Return Selected**: you cannot return more than has been issued for that line.
      The backend calculates already-returned quantity and trims the requested amount accordingly.
    """
    from extensions import db
    from models import IssuedPartRecord, IssuedBatch, Part
    from datetime import datetime, time as _time
    from sqlalchemy import func

    # ---------- helpers ----------
    def _is_return_row(r):
        """A row is a 'return' when its quantity is negative."""
        return (getattr(r, 'quantity', 0) or 0) < 0

    def _next_invoice_number():
        """Safe next invoice number from max(IssuedBatch, IssuedPartRecord)."""
        mb = db.session.query(func.coalesce(func.max(IssuedBatch.invoice_number), 0)).scalar() or 0
        ml = db.session.query(func.coalesce(func.max(IssuedPartRecord.invoice_number), 0)).scalar() or 0
        return max(int(mb), int(ml)) + 1

    def _ensure_invoice_number_for_records(
        records,
        issued_to,
        issued_by,
        reference_job,
        issue_date,
        location,
        force_new=False
    ):
        """
        Safe version: when force_new=True — always creates a new batch with a unique invoice_number.
        Used for repeated returns / extra issues.
        """
        from extensions import db
        from models import IssuedBatch

        if not records:
            return None

        # If not forced and at least one record already has a number — do nothing
        if not force_new and any(getattr(r, "invoice_number", None) for r in records):
            return None

        # Primary path — your helper
        try:
            batch = _create_batch_for_records(
                records=records,
                issued_to=issued_to,
                issued_by=issued_by,
                reference_job=reference_job,
                issue_date=issue_date,
                location=location,
            )
            return batch
        except Exception:
            db.session.rollback()

        # Fallback — reserve manually
        for _ in range(5):
            inv_no = _next_invoice_number()
            try:
                with db.session.begin_nested():
                    batch = IssuedBatch(
                        invoice_number=inv_no,
                        issued_to=issued_to,
                        issued_by=issued_by or "system",
                        reference_job=reference_job,
                        issue_date=issue_date,
                        location=(location or None),
                    )
                    db.session.add(batch)
                    db.session.flush()

                    for r in records:
                        r.batch_id = batch.id
                        r.invoice_number = inv_no
                    db.session.flush()
                return batch
            except Exception:
                db.session.rollback()
        raise RuntimeError("Failed to reserve invoice number")

    def _already_returned_qty_for(r) -> int:
        """
        How many units have already been returned against the same *issued* row r.
        """
        if r is None or (r.quantity or 0) <= 0 or not getattr(r, "part_id", None):
            return 0

        part_id = r.part_id
        issued_to = (r.issued_to or '')
        ref_orig = (r.reference_job or '').strip()

        base = (db.session.query(func.coalesce(func.sum(IssuedPartRecord.quantity), 0))
                .filter(IssuedPartRecord.part_id == part_id,
                        IssuedPartRecord.issued_to == issued_to,
                        IssuedPartRecord.quantity < 0))
        if ref_orig:
            base = base.filter(IssuedPartRecord.reference_job == f"RETURN {ref_orig}")
        else:
            base = base.filter(IssuedPartRecord.reference_job.ilike("RETURN%"))

        total_neg = base.scalar() or 0
        return abs(int(total_neg))

    def _available_to_return_for(r) -> int:
        """Remaining quantity that can still be returned for this issued row."""
        issued_qty = int(r.quantity or 0)
        already_ret = _already_returned_qty_for(r)
        return max(0, issued_qty - already_ret)

    # ---------- access ----------
    if current_user.role not in ['superadmin', 'admin']:
        flash("Access denied", "danger")
        return redirect(url_for('inventory.reports_grouped'))

    # ---------- flags / scope ----------
    do_return        = (request.form.get('do_return') or '') == '1'
    do_delete_ret    = (request.form.get('delete_return') or '') == '1'
    delete_invoice_s = (request.form.get('delete_invoice') or '').strip()
    delete_invoice_no = int(delete_invoice_s) if delete_invoice_s.isdigit() else None
    apply_scope = (request.form.get('apply_scope') or 'all').lower()  # 'all' | 'selected'

    # ---------- header values ----------
    location  = (request.form.get('location') or '').strip() or None
    issued_by = (request.form.get('issued_by') or getattr(current_user, 'username', '') or '').strip()

    s_date = (request.form.get('issue_date') or request.form.get('issue_date_old') or '').strip()
    issue_date = _parse_dt_flex(s_date) if s_date else datetime.utcnow()

    invoice_s = (request.form.get('invoice_number') or '').strip()
    form_invoice_no = int(invoice_s) if invoice_s.isdigit() else None

    # Group keys for scope='all' when invoice number is missing
    g_issued_to = (request.form.get('group_issued_to') or '').strip()
    g_ref_job   = (request.form.get('group_reference_job') or '').strip() or None
    g_issued_by = (request.form.get('group_issued_by') or '').strip()
    g_issue_s   = (request.form.get('group_issue_date') or '').strip()

    # ---------- collect selected ids ----------
    raw_ids = request.form.getlist('record_ids[]') or request.form.getlist('record_ids')
    sel_ids = [int(x) for x in raw_ids if str(x).strip().isdigit()]

    # ---------- load records ----------
    recs = []
    if apply_scope == 'selected' and sel_ids:
        recs = IssuedPartRecord.query.filter(IssuedPartRecord.id.in_(sel_ids)).all()

    if not recs and form_invoice_no is not None:
        recs = IssuedPartRecord.query.filter_by(invoice_number=form_invoice_no).all()

    if not recs and apply_scope == 'all':
        # When invoice has no number yet — resolve by legacy keys within the day
        if g_issued_to and g_issued_by and g_issue_s:
            dt = _parse_dt_flex(g_issue_s) or issue_date
            start = datetime.combine(dt.date(), _time.min)
            end   = datetime.combine(dt.date(), _time.max)
            recs = (IssuedPartRecord.query
                    .filter(IssuedPartRecord.issued_to == g_issued_to,
                            IssuedPartRecord.issued_by == g_issued_by,
                            IssuedPartRecord.reference_job == g_ref_job,
                            IssuedPartRecord.issue_date.between(start, end))
                    .order_by(IssuedPartRecord.id.asc()).all())

    if not recs:
        flash("Invoice lines not found.", "warning")
        return redirect(url_for('inventory.reports_grouped'))

    # ---------- DELETE INVOICE ----------
    if delete_invoice_no is not None:
        if current_user.role != 'superadmin':
            flash("Only superadmin may delete invoice.", "danger")
            return redirect(url_for('inventory.reports_grouped'))
        touched = set()
        try:
            for r in list(recs):
                if r.part:
                    # Revert stock impact (return issued qty to stock; for negative rows this subtracts)
                    r.part.quantity = int(r.part.quantity or 0) + int(r.quantity or 0)
                if r.batch_id:
                    touched.add(r.batch_id)
                db.session.delete(r)
            # Remove empty batches
            for bid in touched:
                still = db.session.query(IssuedPartRecord.id).filter_by(batch_id=bid).first()
                if not still:
                    b = db.session.get(IssuedBatch, bid)
                    if b:
                        db.session.delete(b)
            db.session.commit()
            flash(f"Invoice #{delete_invoice_no:06d} deleted.", "success")
        except Exception as e:
            db.session.rollback()
            flash(f"Failed to delete invoice: {e}", "danger")
        return redirect(url_for('inventory.reports_grouped'))

    # ---------- DELETE RETURN ----------
    if do_delete_ret:
        if current_user.role != 'superadmin':
            flash("Only superadmin may delete return invoices.", "danger")
            return redirect(url_for('inventory.reports_grouped'))
        touched = set()
        try:
            for r in recs:
                if not _is_return_row(r):
                    continue
                if r.part and (r.quantity or 0) < 0:
                    # Deleting a return (negative) reduces stock back
                    r.part.quantity = int(r.part.quantity or 0) - abs(int(r.quantity or 0))
                if r.batch_id:
                    touched.add(r.batch_id)
                db.session.delete(r)
            for bid in list(touched):
                still = db.session.query(IssuedPartRecord.id).filter_by(batch_id=bid).first()
                if not still:
                    b = db.session.get(IssuedBatch, bid)
                    if b:
                        db.session.delete(b)
            db.session.commit()
            flash("Return lines deleted.", "success")
        except Exception as e:
            db.session.rollback()
            flash(f"Failed to delete return invoice: {e}", "danger")
        return redirect(url_for('inventory.reports_grouped'))

    # ---------- RETURN SELECTED (with hard cap) ----------
    if do_return:
        try:
            created = 0
            trimmed_any = False

            for r in recs:
                # Only process against a positive "issued" row
                if (r.quantity or 0) <= 0:
                    continue

                available = _available_to_return_for(r)
                if available <= 0:
                    continue

                raw = (request.form.get(f"qty_{r.id}") or "1").strip()
                try:
                    qty_req = int(raw)
                except Exception:
                    qty_req = 1

                if qty_req < 0:
                    qty_req = 0

                if qty_req > available:
                    qty_req = available
                    trimmed_any = True

                if qty_req <= 0:
                    continue

                # Create a negative "return" row
                ret = IssuedPartRecord(
                    part_id=r.part_id,
                    issued_to=r.issued_to,
                    issued_by=issued_by or r.issued_by,
                    quantity=-qty_req,
                    unit_cost_at_issue=(
                        r.unit_cost_at_issue
                        if r.unit_cost_at_issue is not None
                        else (r.part.unit_cost if r.part else 0.0)
                    ),
                    reference_job=(
                        f"RETURN {r.reference_job}"
                        if (r.reference_job and not str(r.reference_job).upper().startswith("RETURN"))
                        else (r.reference_job or "RETURN")
                    ),
                    issue_date=issue_date,
                    location=(location or r.location),
                    invoice_number=None,
                    batch_id=None
                )
                db.session.add(ret)

                if r.part:
                    r.part.quantity = int(r.part.quantity or 0) + qty_req

                created += 1

            # assign a fresh invoice number for the newly created return rows
            if created:
                new_returns = (
                    db.session.query(IssuedPartRecord)
                    .filter(IssuedPartRecord.invoice_number.is_(None))
                    .filter(IssuedPartRecord.quantity < 0)
                    .order_by(IssuedPartRecord.id.desc())
                    .limit(created)
                    .all()
                )
                if new_returns:
                    return_ref = getattr(new_returns[0], "reference_job", None) or "RETURN"
                    _ensure_invoice_number_for_records(
                        records=new_returns,
                        issued_to=getattr(new_returns[0], "issued_to", None),
                        issued_by=issued_by,
                        reference_job=return_ref,
                        issue_date=datetime.utcnow(),
                        location=getattr(new_returns[0], "location", None) or location,
                        force_new=True
                    )

            db.session.commit()

            if created:
                note = " (some requested quantities were trimmed)" if trimmed_any else ""
                flash(f"Created {created} return line(s){note}.", "success")
            else:
                flash("No lines were eligible for return.", "warning")

        except Exception as e:
            db.session.rollback()
            flash(f"Failed to create return: {e}", "danger")

        return redirect(url_for('inventory.reports_grouped'))

    # ---------- SAVE (edit + assign invoice number for scope='all') ----------
    try:
        # Superadmin can edit line fields with proper stock adjustment
        if current_user.role == 'superadmin':
            for r in recs:
                # Qty edit
                new_qty_s = request.form.get(f"edit_qty_{r.id}")
                if new_qty_s not in (None, ""):
                    try:
                        new_qty = int(new_qty_s)
                        old_qty = int(r.quantity or 0)
                        if r.part:
                            # Put back the difference: old - new
                            r.part.quantity = int(r.part.quantity or 0) + (old_qty - new_qty)
                        r.quantity = new_qty
                    except Exception:
                        pass

                # Unit cost edit
                new_ucost_s = request.form.get(f"edit_ucost_{r.id}")
                if new_ucost_s not in (None, ""):
                    try:
                        r.unit_cost_at_issue = float(new_ucost_s)
                    except Exception:
                        pass

                # Issued To / Reference Job edit
                new_it = request.form.get(f"edit_issued_to_{r.id}")
                if new_it is not None:
                    r.issued_to = new_it.strip()
                new_rj = request.form.get(f"edit_refjob_{r.id}")
                if new_rj is not None:
                    r.reference_job = (new_rj.strip() or None)

        # Common header fields
        for r in recs:
            r.issued_by = issued_by or r.issued_by
            r.issue_date = issue_date or r.issue_date
            if location:
                r.location = location

        # Assign invoice number only when saving the whole card and only if none of the rows has a number
        if apply_scope == 'all' and all(getattr(r, "invoice_number", None) is None for r in recs):
            base = recs[0]
            _ensure_invoice_number_for_records(
                records=recs,
                issued_to=getattr(base, "issued_to", ""),
                issued_by=getattr(base, "issued_by", ""),
                reference_job=getattr(base, "reference_job", None),
                issue_date=getattr(base, "issue_date", issue_date),
                location=getattr(base, "location", location),
            )

        db.session.commit()
        flash("Invoice saved.", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"Failed to save invoice: {e}", "danger")

    return redirect(url_for('inventory.reports_grouped'))

@inventory_bp.route('/reports/cancel_invoice', methods=['POST'])
@login_required
def cancel_invoice():
    if current_user.role not in ['superadmin', 'admin']:
        flash("Access denied", "danger")
        return redirect(url_for('inventory.reports'))

    ids = request.form.getlist('record_ids[]') or request.form.getlist('record_ids')
    records = []
    if ids:
        try:
            ids = [int(x) for x in ids]
        except ValueError:
            ids = []
        if ids:
            records = IssuedPartRecord.query.filter(IssuedPartRecord.id.in_(ids)).all()

    if not records:
        issued_to  = (request.form.get('issued_to') or request.args.get('issued_to') or '').strip()
        reference  = (request.form.get('reference_job') or request.args.get('reference_job') or '').strip() or None
        issued_by  = (request.form.get('issued_by') or request.args.get('issued_by') or '').strip()
        s          = (request.form.get('issue_date') or request.args.get('issue_date') or '').strip()
        issue_date = _parse_dt_flex(s)

        if not (issued_to and issued_by and issue_date):
            flash("Invoice identifiers are missing.", "danger")
            return redirect(url_for('inventory.reports'))

        start = datetime.combine(issue_date.date(), time.min)
        end   = datetime.combine(issue_date.date(), time.max)
        records = IssuedPartRecord.query.filter(
            IssuedPartRecord.issued_to == issued_to,
            IssuedPartRecord.reference_job == reference,
            IssuedPartRecord.issued_by == issued_by,
            IssuedPartRecord.issue_date.between(start, end)
        ).all()

    if not records:
        flash("Invoice not found.", "warning")
        return redirect(url_for('inventory.reports'))

    for r in records:
        part = Part.query.get(r.part_id)
        if part:
            part.quantity = int(part.quantity or 0) + int(r.quantity or 0)
        db.session.delete(r)
    db.session.commit()

    flash("Invoice canceled and stock restored.", "success")
    return redirect(url_for('inventory.reports'))

# ----------------- Download Invoice -----------------
@inventory_bp.route("/import", methods=["GET", "POST"])
@login_required
def import_parts():
    """
    Загрузка PDF → парсинг → превью.
    Безопасно хранит путь к файлу в meta['attachment_path'].
    """
    import os
    from datetime import datetime
    from flask import current_app, request, redirect, url_for, render_template, flash
    from werkzeug.utils import secure_filename

    if request.method == "GET":
        return render_template("receiving_import_upload.html")

    # ---------- POST ----------
    file = request.files.get("file")
    if not file or not file.filename.strip():
        flash("Select a PDF file.", "warning")
        return redirect(url_for("inventory.import_parts"))

    src_name = secure_filename(file.filename)               # исходное имя файла
    ext = os.path.splitext(src_name)[1].lower()
    if ext != ".pdf":
        flash("Only PDF is supported.", "warning")
        return redirect(url_for("inventory.import_parts"))

    # Куда сохраняем
    upload_dir = os.path.join(current_app.instance_path, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    pdf_path = os.path.join(upload_dir, f"{datetime.utcnow():%Y%m%d_%H%M%S}_{src_name}")

    # Сохраняем файл
    file.save(pdf_path)

    # Парсим PDF → DataFrame → нормализация строк
    try_ocr = bool(request.form.get("try_ocr"))
    df = dataframe_from_pdf(pdf_path, try_ocr=try_ocr)      # твой хелпер
    rows = _norm_cols(df) if not df.empty else []
    if not rows:
        flash("Nothing parsed from PDF. Try OCR option.", "warning")
        return redirect(url_for("inventory.import_parts"))

    # Метаданные заголовка (можно править на превью)
    supplier = (request.form.get("supplier") or rows[0].get("supplier") or "").strip()
    invoice  = (request.form.get("invoice_number") or "").strip()
    date_s   = (request.form.get("invoice_date") or "").strip()
    notes    = (request.form.get("notes") or f"Imported from {src_name}").strip()

    # Рендер превью, ПЕРЕДАЁМ путь скрытым полем
    return render_template(
        "receiving_import_preview.html",
        rows=rows,
        meta=dict(
            supplier=supplier,
            invoice=invoice,
            date=date_s,
            notes=notes,
            currency="USD",
            attachment_path=pdf_path,        # ← тут путь для hidden
        ),
        src_file=src_name
    )


@inventory_bp.route('/invoice/<int:record_id>')
@login_required
def invoice(record_id):
    record = IssuedPartRecord.query.get(record_id)
    if not record:
        return "Record not found", 404

    pdf_data = generate_invoice_pdf(record)
    return send_file(BytesIO(pdf_data), as_attachment=True, download_name='invoice.pdf')


ALLOWED_EXTS = {".xlsx", ".xls", ".csv", ".pdf"}

def allowed_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTS


@inventory_bp.route('/reports/download')
@login_required
def download_report_pdf():
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, landscape
    from reportlab.lib.styles import getSampleStyleSheet
    from datetime import datetime
    from io import BytesIO

    start = request.args.get('start_date')
    end = request.args.get('end_date')
    recipient = request.args.get('recipient')
    reference_job = request.args.get('reference_job')

    query = IssuedPartRecord.query.join(Part)
    if start:
        query = query.filter(IssuedPartRecord.issue_date >= start)
    if end:
        end_dt = datetime.strptime(end, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
        query = query.filter(IssuedPartRecord.issue_date <= end_dt)
    if recipient:
        query = query.filter(IssuedPartRecord.issued_to.ilike(f"%{recipient}%"))
    if reference_job:
        query = query.filter(IssuedPartRecord.reference_job.ilike(f"%{reference_job}%"))

    records = query.order_by(IssuedPartRecord.issue_date.desc()).all()

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(letter), leftMargin=30, rightMargin=30, topMargin=30, bottomMargin=20)
    elements = []
    styles = getSampleStyleSheet()

    title = Paragraph("Issued Parts Report", styles["Heading1"])
    elements.append(title)
    elements.append(Spacer(1, 12))

    filter_text = ""
    if start and end:
        filter_text = f"Filters: from {datetime.strptime(start, '%Y-%m-%d').strftime('%m/%d/%Y')} to {datetime.strptime(end, '%Y-%m-%d').strftime('%m/%d/%Y')}"
    elif start:
        filter_text = f"Filters: from {datetime.strptime(start, '%Y-%m-%d').strftime('%m/%d/%Y')}"
    elif end:
        filter_text = f"Filters: up to {datetime.strptime(end, '%Y-%m-%d').strftime('%m/%d/%Y')}"

    if filter_text:
        elements.append(Paragraph(filter_text, styles["Normal"]))
        elements.append(Spacer(1, 12))

    data = [["Date", "Part #", "Name", "Qty", "Unit Cost", "Total", "Issued To", "Job Ref."]]

    total_sum = 0
    for r in records:
        total = r.quantity * r.part.unit_cost
        total_sum += total
        row = [
            r.issue_date.strftime('%m/%d/%Y'),
            Paragraph(r.part.part_number, styles['Normal']),
            Paragraph(r.part.name, styles['Normal']),
            str(r.quantity),
            f"${r.unit_cost_at_issue:.2f}",
            f"${total:.2f}",
            Paragraph(r.issued_to, styles['Normal']),
            Paragraph(r.reference_job or '—', styles['Normal'])
        ]
        data.append(row)

    # Итоговая строка
    data.append([
        "", "", "", "", "TOTAL:",
        f"${total_sum:.2f}", "", ""
    ])

    col_widths = [65, 130, 150, 35, 65, 65, 90, 110]

    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#d3d3d3")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (3, 1), (5, -2), 'RIGHT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),

        # Здесь рисуем сетку по всей таблице
        ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),

        # Вертикальная линия слева от колонки "Unit Cost"
        ('LINEBEFORE', (5, 0), (5, -2), 0.25, colors.grey),

        # Линия над итоговой строкой
        ('LINEABOVE', (4, -1), (5, -1), 0.25, colors.grey),
        # Итоговая строка - светлый фон, отступы, выравнивание
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor("#f0f0f0")),  # светло-серый фон
        ('TOPPADDING', (0, -1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, -1), (-1, -1), 8),

        # Итоговые шрифты
        ('FONTNAME', (4, -1), (4, -1), 'Helvetica-Bold'),
        ('FONTNAME', (5, -1), (5, -1), 'Helvetica-Bold'),

        # Выравнивание итога по правому краю
        ('ALIGN', (5, -1), (5, -1), 'RIGHT'),


    ]))

    elements.append(table)
    doc.build(elements)

    buffer.seek(0)
    return send_file(buffer,
                     as_attachment=True,
                     download_name=f"report_{start or 'all'}_{end or 'all'}.pdf",
                     mimetype='application/pdf')


@inventory_bp.route("/users", methods=["GET", "POST"])
@login_required
def users():
    # ---- Создание нового пользователя (разрешено только superadmin)
    if request.method == "POST":
        if current_user.role != ROLE_SUPERADMIN:
            flash("Access denied", "danger")
            return redirect(url_for("inventory.users"))

        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""
        role     = (request.form.get("role") or "").strip().lower()

        if not username or not password:
            flash("Username and password are required.", "warning")
            return redirect(url_for("inventory.users"))

        # case-insensitive проверка уникальности
        exists = (
            db.session.query(User.id)
            .filter(func.lower(User.username) == username.lower())
            .first()
        )
        if exists:
            flash("User already exists.", "warning")
            return redirect(url_for("inventory.users"))

        # модель сама нормализует роль (валидатор), дефолт — technician
        u = User(username=username, role=role or ROLE_TECHNICIAN)
        u.password = password
        db.session.add(u)
        db.session.commit()
        flash("User created.", "success")
        return redirect(url_for("inventory.users"))

    # ---- Список пользователей
    users = User.query.order_by(User.username.asc()).all()

    # Опции ролей для селекта
    role_options = [
        ("technician", "Technician"),
        ("user",       "User"),
        ("viewer",     "Viewer"),
        ("admin",      "Admin"),
        ("superadmin", "Superadmin"),
    ]

    return render_template("users.html", users=users, role_options=role_options)
@inventory_bp.route('/users/add', methods=['GET', 'POST'])
@login_required
def add_user():
    if current_user.role not in ['superadmin', 'admin']:
        flash("Access denied", "danger")
        return redirect(url_for('inventory.dashboard'))

    if request.method == 'POST':
        username = request.form['username'].strip()
        role = request.form['role']
        password = request.form['password']

        # Админ может создавать только user
        if current_user.role == 'admin' and role != 'user':
            flash("Admins can only create users with role 'user'.", "danger")
            return redirect(url_for('inventory.add_user'))

        if User.query.filter_by(username=username).first():
            flash("Username already exists", "danger")
            return redirect(url_for('inventory.add_user'))

        new_user = User(
            username=username,
            role=role,
            password_hash=generate_password_hash(password)
        )
        db.session.add(new_user)
        db.session.commit()
        flash("User added successfully", "success")
        return redirect(url_for('inventory.users'))

    allowed_roles = ['user'] if current_user.role == 'admin' else ['user', 'admin', 'superadmin']
    return render_template('add_user.html', allowed_roles=allowed_roles)


@inventory_bp.route('/users/edit/<int:user_id>', methods=['GET', 'POST'])
@login_required
def edit_user(user_id):
    from sqlalchemy import func
    user = User.query.get_or_404(user_id)

    me_role = (current_user.role or '').lower()
    target_role = (user.role or '').lower()

    # Права: superadmin — всех; admin — только user/viewer/technician и себя
    if me_role == 'admin':
        if not (user.id == current_user.id or target_role in {'user', 'viewer', 'technician'}):
            flash("Admins can only edit users with role 'user/viewer/technician' or themselves.", "danger")
            return redirect(url_for('inventory.dashboard'))
    elif me_role != 'superadmin':
        flash("Access denied", "danger")
        return redirect(url_for('inventory.dashboard'))

    # Список ролей для селекта:
    # - супер-админ видит все роли
    # - админ может назначать только user/viewer/technician; себе — оставлять admin
    role_options = []
    if me_role == 'superadmin':
        role_options = [
            ('technician', 'Technician'),
            ('user', 'User'),
            ('viewer', 'Viewer'),
            ('admin', 'Admin'),
            ('superadmin', 'Superadmin'),
        ]
    else:  # admin
        role_options = [
            ('technician', 'Technician'),
            ('user', 'User'),
            ('viewer', 'Viewer'),
        ]
        # если админ редактирует себя — оставляем возможность иметь admin (но не менять на супер-админа)
        if user.id == current_user.id:
            role_options.append(('admin', 'Admin'))

    if request.method == 'POST':
        new_username = (request.form.get('username') or '').strip()
        new_role_raw = (request.form.get('role') or '').strip().lower()

        # username можно менять обоим (как было), без усложнения логики
        if new_username:
            # защита от дублей по нижнему регистру
            exists = (
                db.session.query(User.id)
                .filter(func.lower(User.username) == new_username.lower(), User.id != user.id)
                .first()
            )
            if exists:
                flash("Username already exists.", "danger")
                return redirect(url_for('inventory.edit_user', user_id=user.id))
            user.username = new_username

        # роль — только в пределах разрешённых опций
        allowed_values = {val for val, _ in role_options}
        if new_role_raw in allowed_values:
            user.role = new_role_raw  # у вас в модели стоит нормализация/дефолт
        else:
            # ничего не меняем, чтобы не сломать существующее значение
            pass

        db.session.commit()
        flash("User updated successfully", "success")
        return redirect(url_for('inventory.users'))

    return render_template('edit_user.html', user=user, role_options=role_options)

@inventory_bp.route('/users/change_password/<int:user_id>', methods=['GET', 'POST'])
@login_required
def change_password(user_id):
    from flask import request, flash, redirect, url_for, render_template
    from flask_login import current_user
    from werkzeug.security import generate_password_hash
    from extensions import db
    from models import User

    user = User.query.get_or_404(user_id)

    role = (current_user.role or '').lower()
    is_self = current_user.id == user_id

    # superadmin → любой; admin → только user и свой; остальные (в т.ч. technician) → только себе
    if role == 'admin':
        if (user.role or '').lower() != 'user' and not is_self:
            flash("Admins can only change passwords for users with role 'user' or themselves.", "danger")
            return redirect(url_for('inventory.dashboard'))
    elif role != 'superadmin' and not is_self:
        flash("Access denied", "danger")
        return redirect(url_for('inventory.dashboard'))

    if request.method == 'POST':
        if is_self and role != 'superadmin':
            current_password = (request.form.get('current_password') or '').strip()
            if not current_password or not user.check_password(current_password):
                flash("Current password is incorrect.", "danger")
                return redirect(url_for('inventory.change_password', user_id=user_id))

        new_password = (request.form.get('password') or '').strip()
        confirm_password = (request.form.get('confirm_password') or '').strip()

        if not new_password:
            flash("New password cannot be empty.", "danger")
            return redirect(url_for('inventory.change_password', user_id=user_id))

        if new_password != confirm_password:
            flash("Passwords do not match.", "danger")
            return redirect(url_for('inventory.change_password', user_id=user_id))

        user.password_hash = generate_password_hash(new_password)
        db.session.commit()
        flash("Password changed successfully", "success")

        return redirect(url_for('inventory.users' if role == 'superadmin' else 'inventory.dashboard'))

    need_current = is_self and role != 'superadmin'
    return render_template('change_password.html', user=user, need_current=need_current)

@inventory_bp.route('/users/delete/<int:user_id>', methods=['POST'])
@login_required
def delete_user(user_id):
    user = User.query.get_or_404(user_id)

    # Только superadmin может удалять
    if current_user.role != 'superadmin':
        flash("Access denied", "danger")
        return redirect(url_for('inventory.dashboard'))

    if user.id == current_user.id:
        flash("You cannot delete yourself!", "danger")
        return redirect(url_for('inventory.users'))

    db.session.delete(user)
    db.session.commit()
    flash("User deleted successfully", "success")
    return redirect(url_for('inventory.users'))


@inventory_bp.route('/clear_issued_records')
@login_required
def clear_issued_records():
    if current_user.role != 'superadmin':
        flash('Only superadmin can clear records.', 'danger')
        return redirect(url_for('inventory.dashboard'))

    from models import IssuedPartRecord
    from extensions import db

    IssuedPartRecord.query.delete()
    db.session.commit()
    flash('All issued records cleared.', 'success')
    return redirect(url_for('inventory.dashboard'))

@inventory_bp.route('/update_part/<int:part_id>', methods=['POST'])
@login_required
def update_part_field(part_id):
    if current_user.role != 'superadmin':
        flash("Access denied", "danger")
        return redirect(url_for('inventory.dashboard'))

    part = Part.query.get_or_404(part_id)

    quantity = request.form.get('quantity')
    unit_cost = request.form.get('unit_cost')

    if quantity is not None:
        try:
            part.quantity = int(quantity)
        except ValueError:
            flash("Invalid quantity value", "danger")
            return redirect(url_for('inventory.dashboard'))

    if unit_cost is not None:
        try:
            part.unit_cost = float(unit_cost)
        except ValueError:
            flash("Invalid unit cost value", "danger")
            return redirect(url_for('inventory.dashboard'))

    db.session.commit()
    flash("Part updated successfully.", "success")
    return redirect(url_for('inventory.dashboard'))

@inventory_bp.route('/delete/<int:part_id>', methods=['POST'])
@login_required
def delete_part(part_id):
    if current_user.role != 'superadmin':
        flash('Only Superadmin can delete parts.', 'danger')
        return redirect(url_for('inventory.dashboard'))

    part = Part.query.get_or_404(part_id)
    db.session.delete(part)
    db.session.commit()
    flash(f'Part {part.part_number} deleted.', 'success')
    return redirect(url_for('inventory.dashboard'))

@inventory_bp.route('/edit/<int:part_id>', methods=['GET', 'POST'])
@login_required
def edit_part(part_id):
    if current_user.role != 'superadmin':
        flash("Access denied", "danger")
        return redirect(url_for('inventory.dashboard'))

    part = Part.query.get_or_404(part_id)

    if request.method == 'POST':
        part.name = request.form['name'].strip().upper()
        part.part_number = request.form['part_number'].strip().upper()
        try:
            part.quantity = int(request.form['quantity'])
            part.unit_cost = float(request.form['unit_cost'])
        except ValueError:
            flash("Invalid quantity or unit cost", "danger")
            return redirect(url_for('inventory.edit_part', part_id=part_id))

        part.location = request.form['location'].strip().upper()

        db.session.commit()
        flash("Part updated successfully.", "success")
        return redirect(url_for('inventory.dashboard'))

    return render_template('edit_part.html', part=part)


@inventory_bp.route('/reports/delete/<int:record_id>', methods=['POST'])
@login_required
def delete_report_record(record_id):
    if current_user.role != 'superadmin':
        flash('Access denied', 'danger')
        return redirect(url_for('inventory.reports'))

    record = IssuedPartRecord.query.get_or_404(record_id)
    db.session.delete(record)
    db.session.commit()
    flash('Issued record deleted successfully.', 'success')
    return redirect(url_for('inventory.reports'))

@inventory_bp.route('/clear_parts')
@login_required
def clear_parts():
    if current_user.role != 'superadmin':
        flash('Only superadmin can clear parts.', 'danger')
        return redirect(url_for('inventory.dashboard'))

    from models import Part
    from extensions import db

    Part.query.delete()
    db.session.commit()
    flash('All parts cleared.', 'success')
    return redirect(url_for('inventory.dashboard'))

@inventory_bp.route('/compare_cart')
def compare_cart():
    try:
        flash("🔍 Collecting Marcone cart...", "info")
        items = get_marcone_items()

        flash("📦 Comparing with inventory...", "info")
        result = check_cart_items(items)

        filepath = os.path.join(UPLOAD_DIR, "marcone_inventory_report.docx")
        export_to_docx(result, filename=filepath)

        flash("✅ Marcone report generated! Click below to download.", "success")
        return redirect(url_for("inventory.dashboard"))

    except Exception as e:
        flash(f"❌ Error (Marcone): {str(e)}", "danger")
        return redirect(url_for("inventory.dashboard"))

# @inventory_bp.route('/download_marcone_report')
# def download_marcone_report():
#     filepath = os.path.join(UPLOAD_DIR, "marcone_inventory_report.docx")
#     if os.path.exists(filepath):
#         return send_file(filepath, as_attachment=True)
#     flash("❌ Marcone report not found!", "danger")
#     return redirect(url_for("inventory.dashboard"))


@inventory_bp.route('/compare_reliable')
def compare_reliable():
    try:
        flash("🔍 Collecting Reliable cart...", "info")
        items = get_reliable_items()

        flash("📦 Comparing with inventory...", "info")
        result = check_cart_items(items)

        filepath = os.path.join(UPLOAD_DIR, "reliable_inventory_report.docx")
        export_to_docx(result, filename=filepath)

        flash("✅ Reliable report generated! Click below to download.", "success")
        return redirect(url_for("inventory.dashboard"))

    except Exception as e:
        flash(f"❌ Error (Reliable): {str(e)}", "danger")
        return redirect(url_for("inventory.dashboard"))

# @inventory_bp.route('/download_reliable_report')
# def download_reliable_report():
#     filepath = os.path.join(UPLOAD_DIR, "reliable_inventory_report.docx")
#     if os.path.exists(filepath):
#         return send_file(filepath, as_attachment=True)
#     flash("❌ Reliable report not found!", "danger")
#     return redirect(url_for("inventory.dashboard"))

@inventory_bp.route('/download_marcone_report')
@login_required
def download_marcone_report():
    from compare_cart.run_compare import get_marcone_items, check_cart_items, export_to_docx

    filepath = os.path.join(UPLOAD_DIR, "marcone_inventory_report.docx")

    # 1. Получаем данные из корзины Marcone
    items = get_marcone_items()

    # 2. Сравниваем с базой данных
    result = check_cart_items(items)

    # 3. Генерируем новый отчет
    export_to_docx(result, filename=filepath)

    # 4. Отдаем файл пользователю
    if os.path.exists(filepath):
        @after_this_request
        def remove_file(response):
            try:
                os.remove(filepath)
            except Exception as e:
                print(f"Failed to delete file: {e}")
            return response

        return send_file(filepath, as_attachment=True)

    flash("❌ Marcone report generation failed!", "danger")
    return redirect(url_for("inventory.dashboard"))


@inventory_bp.route('/download_reliable_report')
@login_required
def download_reliable_report():
    from compare_cart.run_compare_reliable import get_reliable_items, check_cart_items, export_to_docx

    filepath = os.path.join(UPLOAD_DIR, "reliable_inventory_report.docx")

    items = get_reliable_items()
    result = check_cart_items(items)
    export_to_docx(result, filename=filepath)

    if os.path.exists(filepath):
        @after_this_request
        def remove_file(response):
            try:
                os.remove(filepath)
            except Exception as e:
                print(f"Failed to delete file: {e}")
            return response

        return send_file(filepath, as_attachment=True)

    flash("❌ Reliable report generation failed!", "danger")
    return redirect(url_for("inventory.dashboard"))

from datetime import datetime, date
import os, logging
from sqlalchemy.exc import IntegrityError

@inventory_bp.route("/import-parts", methods=["GET", "POST"], endpoint="import_parts_upload")
def import_parts_upload():
    import os
    import re
    import logging
    from datetime import datetime, date
    from sqlalchemy.exc import IntegrityError

    # ===== helpers ============================================================
    def _supplier_to_default_location(supplier_hint: str | None) -> str:
        """По имени/подсказке поставщика выбрать дефолтную локацию (по подстроке)."""
        s = (supplier_hint or "").lower()
        for key, loc in SUPPLIER_LOC_DEFAULTS.items():
            if key in s:
                return loc
        return "MAIN"

    def _pick_field(model, candidates):
        for f in candidates:
            if hasattr(model, f):
                return f
        return None

    def _coerce_norm_df(df):
        import pandas as pd
        if df is None:
            return None
        if isinstance(df, pd.DataFrame):
            return df
        try:
            return pd.DataFrame(df)
        except Exception:
            return None

    def _infer_invoice_number(norm_df, source_file: str, supplier_hint: str | None) -> str:
        """
        Пытаемся вытащить человеческий номер инвойса:
        1) смотрим известные колонки из нормализованных данных,
        2) длинные числовые куски из имени файла,
        3) "Invoice ..." в первых страницах PDF,
        иначе "".
        """
        # 1) из DF
        if norm_df is not None:
            for col in ("invoice_no", "invoice", "order_no", "ref", "reference"):
                if col in norm_df.columns:
                    vals = [str(x).strip() for x in norm_df[col].dropna().astype(str).tolist()]
                    vals = [v for v in vals if v and v.lower() not in ("nan", "none", "null")]
                    if vals:
                        from collections import Counter
                        c = Counter(vals)
                        best = c.most_common(1)[0][0]
                        m = re.search(r"\b\d{6,}\b", best.replace(" ", ""))
                        if m:
                            return m.group(0)
                        return best

        # 2) имя файла
        base = os.path.basename(source_file or "")
        m2 = re.findall(r"\d{6,}", base)
        if m2:
            return max(m2, key=len)

        # 3) текст PDF
        if str(source_file).lower().endswith(".pdf"):
            text = ""
            try:
                import fitz  # PyMuPDF
                with fitz.open(source_file) as d:
                    for i in range(min(3, d.page_count)):
                        text += d[i].get_text() + "\n"
            except Exception:
                try:
                    from pdfminer.high_level import extract_text
                    text = extract_text(source_file) or ""
                except Exception:
                    text = ""
            if text:
                mm = re.search(r"Invoice[^0-9]*([0-9]{6,})", text, flags=re.I)
                if mm:
                    return mm.group(1)

        return ""

    def _ensure_norm_columns(df, default_loc: str, saved_path: str):
        """
        Гарантирует обязательные колонки и аккуратно заполняет только ПУСТЫЕ значения.
        НИЧЕГО не перезаписывает, если пользователь уже отредактировал ячейку.
        FIXED: используем .str.strip() вместо .strip() на Series.
        """
        import pandas as pd
        import os

        if df is None:
            cols = [
                "part_number", "part_name", "qty", "quantity", "unit_cost",
                "location", "row_key", "source_file", "supplier",
                "order_no", "invoice_no", "date"
            ]
            return pd.DataFrame(columns=cols)

        df = df.copy()

        # гарантируем обязательные колонки
        need_cols = [
            "part_number", "part_name", "qty", "quantity", "unit_cost", "location",
            "row_key", "source_file", "supplier", "order_no", "invoice_no", "date"
        ]
        for c in need_cols:
            if c not in df.columns:
                df[c] = None

        # qty / quantity синхронизируем
        qty = pd.to_numeric(df["qty"], errors="coerce")
        quantity = pd.to_numeric(df["quantity"], errors="coerce")

        df.loc[qty.isna() & quantity.notna(), "qty"] = quantity
        df.loc[quantity.isna() & qty.notna(), "quantity"] = qty

        df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0).astype(int)
        df.loc[df["qty"] < 0, "qty"] = 0
        df["quantity"] = df["qty"].astype(int)

        # unit_cost -> float (NaN оставляем как NaN)
        df["unit_cost"] = pd.to_numeric(df["unit_cost"], errors="coerce")

        # source_file — только те где пусто
        sf_empty = df["source_file"].isna() | (df["source_file"].astype(str).str.strip() == "")
        df.loc[sf_empty, "source_file"] = saved_path

        # location — только пустые → default_loc; потом нормализуем .str
        loc_empty = df["location"].isna() | (df["location"].astype(str).str.strip() == "")
        df.loc[loc_empty, "location"] = (default_loc or "MAIN")

        df["location"] = (
            df["location"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        # строковые служебные поля
        for col in ("part_number", "part_name", "supplier", "order_no", "invoice_no", "row_key", "source_file"):
            df[col] = (
                df[col]
                .astype(str)
                .replace({"None": ""})
                .fillna("")
                .str.strip()
            )

        # row_key — только там где ПУСТО
        rk_empty = df["row_key"].isna() | (df["row_key"].astype(str).str.strip() == "")
        if rk_empty.any():
            # делаем устойчиво-уникальный ключ на строку файла
            df = df.reset_index(drop=False).rename(columns={"index": "__row_i"})
            file_id = os.path.basename(str(saved_path or ""))

            def _mk_key(row):
                pn = str(row.get("part_number", "")).strip().upper()
                loc = str(row.get("location", "")).strip().upper()
                try:
                    qty_local = int(pd.to_numeric(row.get("qty", 0), errors="coerce") or 0)
                except Exception:
                    qty_local = 0
                cost = row.get("unit_cost", None)
                if pd.isna(cost):
                    cost = "NA"
                else:
                    try:
                        cost = float(cost)
                    except Exception:
                        cost = "NA"
                i = int(row.get("__row_i", 0) or 0)
                return f"{file_id}|{i}|{pn}|{loc}|{qty_local}|{cost}"

            df.loc[rk_empty, "row_key"] = df[rk_empty].apply(_mk_key, axis=1)

            if "__row_i" in df.columns:
                del df["__row_i"]

        # подчистить полностью пустые строки
        drop_mask = (
                (df["part_number"].astype(str).str.strip() == "") &
                (df["part_name"].astype(str).str.strip() == "") &
                (df["qty"] == 0) &
                (df["unit_cost"].fillna(0) == 0)
        )
        if drop_mask.any():
            df = df[~drop_mask].copy()

        return df

    def _detect_supplier_from_content(saved_path: str, df_hint):
        """
        Пытаемся угадать поставщика по имени файла, содержимому таблицы,
        или тексту первых страниц PDF.
        """
        try:
            # 0) имя файла
            base = (saved_path or "").lower()
            if "reliable" in base:
                return "Reliable Parts"
            if "marcone" in base or "marcon" in base:
                return "Marcone"

            # 1) из DataFrame (заголовки и первые значения)
            blob = ""
            if df_hint is not None:
                try:
                    blob = " ".join(df_hint.columns.astype(str))
                except Exception:
                    pass
                try:
                    blob += " " + " ".join(
                        [str(x) for x in df_hint.head(30).astype(str).values.ravel()]
                    )
                except Exception:
                    pass

            blob_l = blob.lower()
            if "reliable" in blob_l:
                return "Reliable Parts"
            if "marcone" in blob_l or "marcon" in blob_l:
                return "Marcone"

            # 2) PDF текст
            text = ""
            try:
                import fitz  # PyMuPDF
                with fitz.open(saved_path) as d:
                    for i in range(min(3, d.page_count)):
                        text += d[i].get_text() + "\n"
            except Exception:
                try:
                    from pdfminer.high_level import extract_text
                    text = extract_text(saved_path) or ""
                except Exception:
                    text = ""

            tl = text.lower()
            if "reliable parts" in tl or "reliable parts inc" in tl:
                return "Reliable Parts"
            if "marcone" in tl or "marcone supply" in tl:
                return "Marcone"

        except Exception:
            pass
        return None

    def _merge_locations_stable(old_loc: str | None, new_loc: str | None) -> str:
        """
        Склеиваем старую и новую локацию без дублей:
        old='C1', new='MAR' -> 'C1/MAR'
        old='C1/MAR', new='MAR' -> 'C1/MAR'
        """
        out = []
        for raw in (old_loc, new_loc):
            if not raw:
                continue
            for token in str(raw).upper().split("/"):
                t = token.strip()
                if t and t not in out:
                    out.append(t)
        return "/".join(out)

    # ==========================================================================

    enabled = int(current_app.config.get("WCCR_IMPORT_ENABLED", 0))
    dry     = int(current_app.config.get("WCCR_IMPORT_DRY_RUN", 1))

    # Для быстрой отладки
    if request.method == "POST":
        keys = list(request.form.keys())
        flash(
            f"DEBUG: method=POST, keys={keys[:12]}{' ...' if len(keys) > 12 else ''}, "
            f"enabled={enabled}, dry={dry}",
            "info"
        )

    # ===== A) POST из превью (Save/Apply) =====================================
    if (
        request.method == "POST" and
        (
            ("save" in request.form) or
            ("apply" in request.form) or
            any(k.startswith("units[") or k.startswith("rows[") for k in request.form.keys())
        )
    ):
        saved_path = (request.form.get("saved_path") or "").strip()
        if not saved_path:
            flash("Saved path is empty. Upload the file again.", "warning")

        # превью-форма -> dict[] -> DataFrame
        rows = parse_preview_rows_relaxed(request.form)
        if not rows:
            flash("Нет данных в форме (пустая таблица).", "warning")
            return render_template("import_preview.html", rows=[], saved_path=saved_path)

        norm = rows_to_norm_df(rows, saved_path)
        norm = _coerce_norm_df(norm)

        # supplier_hint: взять из hidden поля или попытаться угадать
        supplier_hint = (
            (request.form.get("supplier_hint") or "").strip()
            or _detect_supplier_from_content(saved_path, norm)
        )
        default_loc   = _supplier_to_default_location(supplier_hint)

        if norm is None or norm.empty:
            flash("Нет данных для применения импорта (пустой набор строк).", "warning")
            return render_template("import_preview.html", rows=[], saved_path=saved_path)

        # привести колонки и заполнить только пустые
        norm = _ensure_norm_columns(norm, default_loc, saved_path)
        flash(
            f"Supplier hint: {supplier_hint or 'None'}, default location: {default_loc}",
            "info"
        )

        # вычисляем invoice заранее
        invoice_guess = _infer_invoice_number(norm, saved_path, supplier_hint) or ""

        # --- SAVE: просто показать превью с уже нормализованными строками
        if "save" in request.form:
            return render_template(
                "import_preview.html",
                rows=norm.to_dict(orient="records"),
                saved_path=saved_path,
                supplier_hint=supplier_hint or ""
            )

        # --- APPLY: реальный импорт в базу
        if "apply" in request.form:
            if dry or not enabled:
                flash("Импорт в режиме предпросмотра (DRY) или отключён конфигом.", "info")
                return render_template(
                    "import_preview.html",
                    rows=rows,
                    saved_path=saved_path,
                    supplier_hint=supplier_hint or "",
                    default_loc=default_loc or "MAIN",
                )

            session = db.session

            # модели ReceivingBatch/ReceivingItem могут иметь разные имена
            try:
                from models import ReceivingBatch as BatchModel
            except Exception:
                try:
                    from models import Receiving as BatchModel
                except Exception:
                    BatchModel = None

            try:
                from models import ReceivingItem as ItemModel
            except Exception:
                try:
                    from models import ReceivingLine as ItemModel
                except Exception:
                    ItemModel = None

            # создаём batch
            batch = None
            batch_id_field = None
            if BatchModel is not None:
                B_SUP   = _pick_field(BatchModel, ["supplier_name","supplier","vendor","provider"])
                B_INV   = _pick_field(BatchModel, ["invoice_number","invoice_no","invoice","number"])
                B_DATE  = _pick_field(BatchModel, ["date","invoice_date","received_date"])
                B_CURR  = _pick_field(BatchModel, ["currency"])
                B_NOTES = _pick_field(BatchModel, ["notes","comment","description"])
                B_STAT  = _pick_field(BatchModel, ["status","state"])
                B_C_AT  = _pick_field(BatchModel, ["created_at","created"])
                B_C_BY  = _pick_field(BatchModel, ["created_by"])
                B_P_AT  = _pick_field(BatchModel, ["posted_at"])
                B_P_BY  = _pick_field(BatchModel, ["posted_by"])
                B_ATTP  = _pick_field(BatchModel, ["attachment_path","attachment"])
                batch_id_field = _pick_field(BatchModel, ["id","batch_id"])

                batch_kwargs = {}
                if B_SUP:   batch_kwargs[B_SUP]   = (supplier_hint or "Unknown")
                if B_INV:   batch_kwargs[B_INV]   = invoice_guess
                if B_DATE:  batch_kwargs[B_DATE]  = date.today()
                if B_CURR:  batch_kwargs[B_CURR]  = "USD"
                if B_NOTES: batch_kwargs[B_NOTES] = f"Imported from {os.path.basename(saved_path)}"
                if B_STAT:  batch_kwargs[B_STAT]  = "new"
                if B_C_AT:  batch_kwargs[B_C_AT]  = datetime.utcnow()
                if B_C_BY:  batch_kwargs[B_C_BY]  = 0
                if B_P_AT:  batch_kwargs[B_P_AT]  = None
                if B_P_BY:  batch_kwargs[B_P_BY]  = None
                if B_ATTP:  batch_kwargs[B_ATTP]  = ""

                try:
                    batch = BatchModel(**batch_kwargs)
                    session.add(batch)
                    session.flush()
                except IntegrityError as e:
                    session.rollback()
                    batch = None
                    logging.exception("DB error while creating receiving batch: %s", e)
                    flash("DB error while creating receiving batch (см. лог). Продолжаю без батча.", "danger")
                except Exception as e:
                    session.rollback()
                    batch = None
                    logging.exception("Failed to create ReceivingBatch: %s", e)
                    flash("Не удалось создать ReceivingBatch. Продолжаю без батча.", "warning")

            # вспомогательная: проверка дубликатов
            def duplicate_exists(rk: str) -> bool:
                return has_key(rk)

            # основной апдейтер склада
            def make_movement(m: dict) -> None:
                """
                Применяем одну строку приходной накладной к складу.
                ЛОКАЦИЯ:
                  - если до прихода on_hand_before > 0 → MERGE "OLD/NEW"
                  - если до прихода on_hand_before == 0 → ставим только NEW
                Без дублей ("C1/MAR/AMAZ"), не затираем старое просто так.
                """
                PartModel = Part

                PN_FIELDS   = ["part_number","number","sku","code","partnum","pn"]
                NAME_FIELDS = ["name","part_name","descr","description","title"]
                QTY_FIELDS  = ["quantity","qty","on_hand","stock","count"]
                LOC_FIELDS  = ["location","bin","shelf","place","loc"]
                COST_FIELDS = ["unit_cost","cost","price","unitprice","last_cost"]
                SUP_FIELDS  = ["supplier","vendor","provider"]

                pn_field   = _pick_field(PartModel, PN_FIELDS)
                name_field = _pick_field(PartModel, NAME_FIELDS)
                qty_field  = _pick_field(PartModel, QTY_FIELDS)
                loc_field  = _pick_field(PartModel, LOC_FIELDS)
                cost_field = _pick_field(PartModel, COST_FIELDS)
                sup_field  = _pick_field(PartModel, SUP_FIELDS)

                if pn_field is None or qty_field is None:
                    raise RuntimeError("Не найдено поле PART # или QTY в модели Part.")

                incoming_pn   = m["part_number"]
                incoming_qty  = int(m.get("qty") or m.get("quantity") or 0)
                incoming_cost = m.get("unit_cost")
                incoming_name = m.get("part_name") or ""
                incoming_loc  = (m.get("location") or "").strip().upper()
                incoming_sup  = m.get("supplier") or ""

                # найти Part только по PN (локация не участвует)
                part = PartModel.query.filter(
                    getattr(PartModel, pn_field) == incoming_pn
                ).first()

                if not part:
                    # создать Part
                    kwargs = {
                        pn_field: incoming_pn,
                        qty_field: 0,  # потом увеличим
                    }
                    if name_field and incoming_name:
                        kwargs[name_field] = incoming_name
                    if loc_field:
                        kwargs[loc_field] = incoming_loc
                    if cost_field and (incoming_cost is not None):
                        kwargs[cost_field] = float(incoming_cost)
                    if sup_field and incoming_sup:
                        kwargs[sup_field] = incoming_sup

                    part = PartModel(**kwargs)
                    session.add(part)
                    try:
                        session.flush()
                    except IntegrityError:
                        session.rollback()
                        part = PartModel.query.filter(
                            getattr(PartModel, pn_field) == incoming_pn
                        ).first()
                        if not part:
                            raise

                # Снимок до пополнения
                on_hand_before = int(getattr(part, qty_field) or 0)
                old_loc_before = getattr(part, loc_field) if loc_field else None

                # Обновляем количество
                setattr(part, qty_field, on_hand_before + incoming_qty)

                # Цена: последний приход побеждает
                if cost_field and (incoming_cost is not None):
                    setattr(part, cost_field, float(incoming_cost))

                # Имя: если раньше было пусто — заполним
                if name_field and incoming_name and not getattr(part, name_field):
                    setattr(part, name_field, incoming_name)

                # Локация: умно, без жёсткой перезаписи
                if loc_field:
                    incoming_loc_up = incoming_loc
                    if incoming_loc_up:
                        if on_hand_before > 0:
                            # был остаток → склеиваем без дублей
                            merged = _merge_locations_stable(old_loc_before, incoming_loc_up)
                            setattr(part, loc_field, merged)
                        else:
                            # до прихода было 0 → просто новая локация
                            setattr(part, loc_field, incoming_loc_up)
                    # если в приходе location пустой -> НЕ трогаем part.location

                # Создаём ReceivingItem (если модель есть и batch создан)
                if ItemModel is not None and batch is not None and batch_id_field is not None:
                    I_BATCH = _pick_field(ItemModel, ["goods_receipt_id","batch_id","receiving_id"])
                    I_PART  = _pick_field(ItemModel, ["part_id","item_part_id"])
                    I_PN    = _pick_field(ItemModel, ["part_number","part","sku","code"])
                    I_PNAME = _pick_field(ItemModel, ["part_name","name","descr","description","title"])
                    I_QTY   = _pick_field(ItemModel, ["qty","quantity"])
                    I_COST  = _pick_field(ItemModel, ["unit_cost","cost","price","unitprice"])
                    I_LOC   = _pick_field(ItemModel, ["location","bin","shelf","place","loc"])

                    try:
                        item_kwargs = {}
                        if I_BATCH:
                            item_kwargs[I_BATCH] = getattr(batch, batch_id_field)
                        if I_PART:
                            item_kwargs[I_PART] = getattr(part, "id", None)
                        else:
                            if I_PN:
                                item_kwargs[I_PN] = incoming_pn
                            if I_PNAME:
                                item_kwargs[I_PNAME] = incoming_name
                        if I_QTY:
                            item_kwargs[I_QTY] = incoming_qty
                        if I_COST and (incoming_cost is not None):
                            item_kwargs[I_COST] = float(incoming_cost)
                        if I_LOC:
                            item_kwargs[I_LOC] = incoming_loc

                        if item_kwargs.get(I_QTY, 0) > 0:
                            session.add(ItemModel(**item_kwargs))
                    except IntegrityError as e:
                        session.rollback()
                        logging.exception("DB error while creating ReceivingItem: %s", e)
                        flash("DB error while creating ReceivingItem (см. лог).", "danger")
                    except Exception as e:
                        session.rollback()
                        logging.exception("Failed to create ReceivingItem: %s", e)
                        flash("Не удалось создать строку приёмки (см. лог).", "warning")

                # Регистрируем ключ от дубликатов (после успешного апдейта)
                meta = {
                    "file": m.get("source_file"),
                    "supplier": (m.get("supplier") or "").strip(),
                    "invoice": (invoice_guess or "").strip(),
                }
                try:
                    if batch is not None and batch_id_field:
                        meta["batch_id"] = getattr(batch, batch_id_field)
                except Exception:
                    pass
                add_key(m["row_key"], meta)

            # строим движения/применяем make_movement к каждой строке
            built, errors = build_receive_movements(
                norm,
                duplicate_exists_func=duplicate_exists,
                make_movement_func=make_movement
            )

            # пытаемся зафиксировать все изменения по складу и строкам
            try:
                session.commit()
            except Exception as e:
                session.rollback()
                logging.exception("Final commit failed: %s", e)
                flash("Ошибка при окончательной записи в базу. Изменения откатились.", "danger")
                return redirect(url_for("inventory.import_parts_upload"))

            # теперь выставляем батчу статус 'posted', но БЕЗ второго плюса на склад
            bid = None
            if batch is not None and batch_id_field:
                try:
                    bid = getattr(batch, batch_id_field or "id", None)

                    # перечитываем объект батча из базы, чтобы он был attach'ed
                    fresh_batch = BatchModel.query.get(bid)
                    if fresh_batch is not None:
                        # найти реальные имена колонок status / posted_at / posted_by
                        B_STAT  = _pick_field(BatchModel, ["status","state"])
                        B_P_AT  = _pick_field(BatchModel, ["posted_at"])
                        B_P_BY  = _pick_field(BatchModel, ["posted_by"])

                        if B_STAT:
                            setattr(fresh_batch, B_STAT, "posted")
                        if B_P_AT:
                            setattr(fresh_batch, B_P_AT, datetime.utcnow())
                        if B_P_BY:
                            # если у тебя есть current_user.id — лучше его, иначе 0
                            try:
                                uid = getattr(current_user, "id", 0)
                            except Exception:
                                uid = 0
                            setattr(fresh_batch, B_P_BY, uid)

                        session.add(fresh_batch)
                        session.commit()

                except Exception as e:
                    session.rollback()
                    logging.exception("Failed to finalize batch status: %s", e)
                    # не фейлим весь процесс, просто скажем что batch остался draft
                    flash("Приход сохранён, но статус партии не удалось установить 'posted'.", "warning")

            # показать ошибки парсинга строк
            for e in errors:
                flash(e, "danger")

            # флешим итог
            if bid is not None:
                flash(f"Stock received and posted. Batch #{bid}. Создано строк: {len(built)}", "success")
                # после успешного импорта ведём на страницу детали партии
                return redirect(url_for("inventory.receiving_detail", batch_id=bid))
            else:
                flash(f"Создано приходов: {len(built)} (без записи ReceivingBatch)", "warning")
                return redirect(url_for("inventory.import_parts_upload"))


    # ===== B) первая загрузка файла → превью ==================================
    if request.method == "POST":
        f = request.files.get("file")
        if not f or not f.filename.strip():
            flash("Выберите файл (.pdf, .xlsx, .xls или .csv)", "warning")
            return redirect(request.url)

        filename   = secure_filename(f.filename)
        upload_dir = current_app.config.get(
            "UPLOAD_FOLDER",
            os.path.join(current_app.instance_path, "uploads")
        )
        os.makedirs(upload_dir, exist_ok=True)
        path = os.path.join(upload_dir, filename)
        f.save(path)

        ext = os.path.splitext(path)[1].lower()
        df  = dataframe_from_pdf(path, try_ocr=False) if ext == ".pdf" else load_table(path)

        # до нормализации — пробуем угадать поставщика
        supplier_hint = _detect_supplier_from_content(path, df)
        default_loc   = _supplier_to_default_location(supplier_hint)
        flash(
            f"Supplier hint: {supplier_hint or 'None'}, default location: {default_loc}",
            "info"
        )

        df = drop_vendor_noise_rows(df)
        df = fix_pn_and_description_in_df(df)

        if df is None or df.empty:
            flash("В этом файле не удалось распознать таблицы (возможно отсканированный PDF).", "danger")
            return render_template("import_preview.html", rows=[], saved_path=path)

        norm, issues = normalize_table(
            df,
            supplier_hint=supplier_hint,
            source_file=path,
            default_location=default_loc,
        )
        for msg in issues:
            flash(msg, "warning")

        # жёсткая гарантия колонок, qty, location и row_key
        norm = _ensure_norm_columns(norm, default_loc, path)

        # подготовка в dict для превью
        rows = norm.to_dict(orient="records")
        rows = fix_norm_records(rows, default_loc)

        return render_template(
            "import_preview.html",
            rows=rows,
            saved_path=path,
            supplier_hint=supplier_hint or ""
        )

    # ===== C) GET → показать форму загрузки ===================================
    return render_template("import_parts.html")

@inventory_bp.get("/orders/", endpoint="list_orders")
def list_orders():
    from flask import request, render_template
    from extensions import db
    # from models.order_item import OrderItem

    q = (request.args.get("q") or "").strip()
    base = OrderItem.query.order_by(OrderItem.date_ordered.desc(), OrderItem.id.desc())
    if q:
        like = f"%{q}%"
        base = base.filter(
            db.or_(
                OrderItem.technician.ilike(like),
                OrderItem.order_number.ilike(like),
            )
        )
    items = base.limit(500).all()
    return render_template("orders_list.html", items=items, q=q)

# ========= ORDERS: отметить как получено (и пополнить склад) =========
@inventory_bp.post("/orders/<int:item_id>/received", endpoint="mark_received")
def mark_received(item_id):
    from flask import redirect, url_for, flash
    from flask_login import current_user
    from datetime import datetime
    from models import db
    from models.order_item import OrderItem
    from models import Part  # твоя существующая модель склада

    # Помощник: подобрать имена полей из твоей модели Part
    PN_FIELDS   = ["part_number","number","sku","code","partnum","pn"]
    QTY_FIELDS  = ["quantity","qty","on_hand","stock","count"]
    LOC_FIELDS  = ["location","bin","shelf","place","loc"]
    NAME_FIELDS = ["name","part_name","descr","description","title"]
    COST_FIELDS = ["unit_cost","cost","price","unitprice","last_cost"]
    SUP_FIELDS  = ["supplier","vendor","provider"]
    def pick_field(model, candidates):
        for f in candidates:
            if hasattr(model, f): return f
        return None

    item = OrderItem.query.get_or_404(item_id)
    if item.status == "received":
        flash("Already received.", "info")
        return redirect(url_for("inventory.list_orders"))

    pn_field   = pick_field(Part, PN_FIELDS)
    qty_field  = pick_field(Part, QTY_FIELDS)
    loc_field  = pick_field(Part, LOC_FIELDS)
    name_field = pick_field(Part, NAME_FIELDS)
    cost_field = pick_field(Part, COST_FIELDS)
    sup_field  = pick_field(Part, SUP_FIELDS)
    if not pn_field or not qty_field:
        flash("Part model missing PN or QTY field.", "danger")
        return redirect(url_for("inventory.list_orders"))

    # Найти/создать позицию на складе (по PN + локация, если есть)
    filters = {pn_field: item.part_number}
    if loc_field and item.location:
        filters[loc_field] = item.location
    part = Part.query.filter_by(**filters).first()
    if not part:
        kwargs = dict(filters)
        kwargs[qty_field] = 0
        if name_field and item.part_name: kwargs[name_field] = item.part_name
        if cost_field and item.unit_cost is not None: kwargs[cost_field] = float(item.unit_cost)
        if sup_field and item.supplier: kwargs[sup_field] = item.supplier
        part = Part(**kwargs)
        db.session.add(part)
        db.session.flush()

    # + остаток
    setattr(part, qty_field, (getattr(part, qty_field) or 0) + int(item.qty_ordered or 0))

    # статус заказа
    item.status = "received"
    item.date_received = datetime.utcnow()
    who = getattr(current_user, "email", "system")
    item.notes = (item.notes or "")
    if who not in (item.notes or ""):
        item.notes = (item.notes + f"\nReceived by {who} at {item.date_received.isoformat()}").strip()

    db.session.commit()
    flash("Marked as received and stock updated.", "success")
    return redirect(url_for("inventory.list_orders"))

@inventory_bp.route("/import-settings", methods=["GET", "POST"], endpoint="import_settings")
@login_required
def import_settings():
    # простая защита: только админ/суперадмин
    if getattr(current_user, "role", None) not in (ROLE_ADMIN, ROLE_SUPERADMIN):
        abort(403)

    if request.method == "POST":
        ocr_on = 1 if request.form.get("pdf_ocr_enabled") == "on" else 0
        set_setting("pdf_ocr_enabled", bool(ocr_on))
        flash("Настройки сохранены.", "success")
        return redirect(url_for("inventory.import_settings"))

    ocr_enabled = bool(get_setting("pdf_ocr_enabled", False))
    return render_template("import_settings.html", ocr_enabled=ocr_enabled)
@inventory_bp.route('/reports/create_return_invoice', methods=['POST'])
@login_required
def create_return_invoice():
    if current_user.role not in ['superadmin', 'admin']:
        flash("Access denied", "danger")
        return redirect(url_for('inventory.reports'))

    # идентификаторы исходной группы (на всякий случай)
    issued_to_old     = (request.form.get('issued_to_old') or request.form.get('issued_to') or '').strip()
    reference_job_old = (request.form.get('reference_job_old') or request.form.get('reference_job') or '').strip() or None
    s                 = (request.form.get('issue_date_old') or request.form.get('issue_date') or '').strip()
    issued_by         = (request.form.get('issued_by') or '').strip()
    issue_date_old    = _parse_dt_flex(s)

    # выбранные строки для возврата
    line_ids = request.form.getlist('line_ids') or request.form.getlist('line_ids[]')
    if not line_ids:
        flash("No lines selected.", "warning")
        return redirect(url_for('inventory.reports'))

    try:
        line_ids = [int(x) for x in line_ids]
    except ValueError:
        flash("Invalid selection.", "danger")
        return redirect(url_for('inventory.reports'))

    # под каждую строку возьмём qty_X
    rows = IssuedPartRecord.query.filter(IssuedPartRecord.id.in_(line_ids)).all()
    if not rows:
        flash("Selected lines not found.", "warning")
        return redirect(url_for('inventory.reports'))

    created = 0
    for r in rows:
        qty_raw = request.form.get(f"qty_{r.id}")
        try:
            qty = int(qty_raw or 0)
        except Exception:
            qty = 0
        qty = max(0, min(qty, int(r.quantity or 0)))  # нельзя вернуть больше, чем выдали
        if qty <= 0:
            continue

        # создаём возвратную запись (отрицательное количество)
        ret = IssuedPartRecord(
            part_id=r.part_id,
            quantity=-qty,
            issued_to=r.issued_to,
            reference_job=f"RETURN {r.reference_job or ''}".strip(),
            issued_by=current_user.username,
            issue_date=datetime.utcnow(),
            unit_cost_at_issue=r.unit_cost_at_issue,
        )
        # при возврате на склад добавим кол-во обратно
        part = Part.query.get(r.part_id)
        if part:
            part.quantity = int(part.quantity or 0) + qty

        db.session.add(ret)
        created += 1

    if created == 0:
        flash("Nothing to return.", "warning")
        return redirect(url_for('inventory.reports'))

    db.session.commit()
    flash(f"Return invoice created ({created} lines).", "success")
    return redirect(url_for('inventory.reports'))

@inventory_bp.route('/reports/delete_return_invoice', methods=['POST'])
@login_required
def delete_return_invoice():
    if current_user.role != 'superadmin':
        flash("Access denied", "danger")
        return redirect(url_for('inventory.reports_grouped'))

    raw_ids = request.form.getlist('record_ids[]') or request.form.getlist('record_ids')
    try:
        ids = [int(x) for x in raw_ids if str(x).strip()]
    except ValueError:
        ids = []

    if not ids:
        flash("Nothing selected to delete.", "warning")
        return redirect(url_for('inventory.reports_grouped'))

    rows = IssuedPartRecord.query.filter(IssuedPartRecord.id.in_(ids)).all()
    if not rows:
        flash("Records not found.", "warning")
        return redirect(url_for('inventory.reports_grouped'))

    touched_batch_ids = set()
    try:
        for r in rows:
            # удаляем только возвратные строки
            if (r.quantity or 0) >= 0 and not ((r.reference_job or '').upper().startswith('RETURN')):
                continue

            # откатываем склад: удаляем возврат → уменьшаем склад
            if r.part and (r.quantity or 0) < 0:
                r.part.quantity = (r.part.quantity or 0) - abs(int(r.quantity or 0))

            if r.batch_id:
                touched_batch_ids.add(r.batch_id)

            db.session.delete(r)

        # чистим пустые батчи
        if touched_batch_ids:
            for bid in list(touched_batch_ids):
                still_has = db.session.query(IssuedPartRecord.id).filter_by(batch_id=bid).first()
                if not still_has:
                    b = db.session.get(IssuedBatch, bid)
                    if b:
                        db.session.delete(b)

        db.session.commit()
        flash("Return invoice deleted.", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"Failed to delete return invoice: {e}", "danger")

    return redirect(url_for('inventory.reports_grouped'))

@inventory_bp.route('/reports/return_selected', methods=['POST'])
@login_required
def return_selected():
    if current_user.role not in ['superadmin', 'admin']:
        flash("Access denied", "danger")
        return redirect(url_for('inventory.reports_grouped'))

    raw_ids = request.form.getlist('record_ids[]') or request.form.getlist('record_ids')
    try:
        selected_ids = [int(x) for x in raw_ids if str(x).strip()]
    except ValueError:
        selected_ids = []

    if not selected_ids:
        flash("No rows selected.", "warning")
        return redirect(url_for('inventory.reports_grouped'))

    rows = IssuedPartRecord.query.filter(IssuedPartRecord.id.in_(selected_ids)).all()
    if not rows:
        flash("Selected records not found.", "warning")
        return redirect(url_for('inventory.reports_grouped'))

    now_dt   = datetime.utcnow()
    issued_by = getattr(current_user, 'username', '') or 'system'
    first = rows[0]
    batch_issued_to = first.issued_to
    batch_reference = (first.reference_job or '').strip() or 'STOCK'
    return_reference = f"RETURN {batch_reference.upper()}"
    batch_location   = first.location

    created = []
    for src in rows:
        # то, что пользователь попросил вернуть
        try:
            want = int(request.form.get(f"qty_{src.id}", "1") or "1")
        except ValueError:
            want = 1
        if want <= 0:
            continue

        # Сколько уже возвращено по этой детали/получателю
        already = _already_returned_qty_for_source(src)

        # Максимум к возврату сейчас = выдано по строке − уже возвращено (не ниже 0)
        issued_qty = int(src.quantity or 0)
        max_can_return = max(0, issued_qty - already)

        qty = min(want, max_can_return)
        if qty == 0:
            continue

        ret = IssuedPartRecord(
            part_id=src.part_id,
            quantity=-qty,
            issued_to=src.issued_to,
            issued_by=issued_by,
            reference_job=return_reference,
            issue_date=now_dt,
            unit_cost_at_issue=src.unit_cost_at_issue,
            location=src.location,
        )
        db.session.add(ret)
        created.append(ret)

        # Возвращаем на склад
        if src.part:
            src.part.quantity = (src.part.quantity or 0) + qty

    if not created:
        flash("Nothing to return.", "warning")
        db.session.rollback()
        return redirect(url_for('inventory.reports_grouped'))

    try:
        batch = create_batch_for_records(
            created,
            issued_to=batch_issued_to,
            issued_by=issued_by,
            reference_job=return_reference,
            issue_date=now_dt,
            location=batch_location,
        )
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        flash(f"Failed to create return invoice: {e}", "danger")
        return redirect(url_for('inventory.reports_grouped'))

    flash(f"Return invoice #{batch.invoice_number} created ({len(created)} lines).", "success")
    return redirect(url_for('inventory.reports_grouped', invoice_number=batch.invoice_number))

def _already_returned_qty_for_source(src) -> int:
    """
    Сколько уже возвращено по той же детали этому же получателю.
    Суммируем все строки с qty < 0, где part_id и issued_to совпадают,
    а reference_job начинается с 'RETURN'.
    """
    total_neg = (
        db.session.query(func.coalesce(func.sum(IssuedPartRecord.quantity), 0))
        .filter(
            IssuedPartRecord.part_id == src.part_id,
            IssuedPartRecord.issued_to == src.issued_to,
            IssuedPartRecord.reference_job.ilike('RETURN%'),
        )
        .scalar()
        or 0
    )
    # total_neg отрицательный или 0 → возвращаем модуль
    return abs(int(total_neg))

@inventory_bp.get("/receiving", endpoint="receiving_list")
@login_required
def receiving_list():
    from sqlalchemy import func
    current_app.logger.debug("### DEBUG receiving_list USING MODEL %s FROM %s TABLENAME=%s", ReceivingBatch, __file__,
                               getattr(ReceivingBatch, "__tablename__", "?"))

    q = ReceivingBatch.query
    supplier = (request.args.get("supplier") or "").strip()
    inv = (request.args.get("invoice") or "").strip()
    d1 = (request.args.get("date_from") or "").strip()
    d2 = (request.args.get("date_to") or "").strip()
    status = (request.args.get("status") or "").strip()

    if supplier:
        q = q.filter(ReceivingBatch.supplier_name.ilike(f"%{supplier}%"))
    if inv:
        q = q.filter(ReceivingBatch.invoice_number.ilike(f"%{inv}%"))
    if status in ("draft", "posted"):
        q = q.filter(ReceivingBatch.status == status)

    def _parse_date(s):
        try:
            return datetime.strptime(s, "%Y-%m-%d").date()
        except Exception:
            return None

    d1p, d2p = _parse_date(d1), _parse_date(d2)
    if d1p:
        q = q.filter(ReceivingBatch.invoice_date >= d1p)
    if d2p:
        q = q.filter(ReceivingBatch.invoice_date <= d2p)

    batches = q.order_by(
        ReceivingBatch.invoice_date.desc().nullslast(),
        ReceivingBatch.id.desc()
    ).all()

    # compute totals per batch (works with both schemas)
    totals = {}
    for b in batches:
        try:
            lines = getattr(b, "items", []) or getattr(b, "lines", []) or []
            total = 0.0
            for ln in lines:
                # flexible attr names
                qty  = getattr(ln, "qty", None) or getattr(ln, "quantity", 0) or 0
                cost = getattr(ln, "unit_cost", None) or getattr(ln, "price", None) or 0.0
                try:
                    total += float(cost) * int(qty)
                except Exception:
                    pass
            totals[b.id] = total
        except Exception:
            totals[b.id] = 0.0

    # pass a superadmin flag; be defensive with attribute names
    can_admin = bool(
        getattr(current_user, "is_superadmin", False) or
        getattr(current_user, "is_super_admin", False) or
        getattr(current_user, "is_admin", False)
    )

    return render_template(
        "receiving_list.html",
        batches=batches,
        totals=totals,
        can_admin=can_admin,
        filters={
            "supplier": supplier, "invoice": inv,
            "date_from": d1, "date_to": d2, "status": status
        }
    )

# --- Receiving: toggle posted/draft (superadmin only) ------------------------
# --- helper: запрещаем unpost, если уже было списание в IssuedPartRecord -----

def _batch_consumed_forbid_unpost(batch) -> bool:
    """
    Возвращает True если из ЭТОГО прихода уже что-то ушло техникам.
    В этом случае unpost НЕЛЬЗЯ делать (иначе сломаешь остатки).

    Логика:
    - соберём все part_number из строк прихода (batch.items / batch.lines)
    - найдём Part.id для этих part_number
    - проверим IssuedPartRecord по этим Part.id
    - если есть хотя бы одна выдача -> True
    """
    from sqlalchemy import func
    from extensions import db
    from models import Part, IssuedPartRecord

    # получаем строки партии
    lines = (
        getattr(batch, "lines", None)
        or getattr(batch, "items", None)
        or []
    )

    # соберём PN из строк
    part_numbers_upper = []
    for it in lines:
        pn = (getattr(it, "part_number", "") or "").strip()
        if pn:
            part_numbers_upper.append(pn.upper())

    # если нет нормальных PN -> считаем что расход не доказан => unpost разрешаем
    if not part_numbers_upper:
        return False

    # найдём ID деталей по PN
    part_ids = [
        row[0]
        for row in (
            db.session.query(Part.id)
            .filter(func.upper(func.trim(Part.part_number)).in_(part_numbers_upper))
            .all()
        )
    ]

    if not part_ids:
        # не нашли Part вообще -> считаем что нечего сверять => не блокируем
        return False

    # ищем хоть одну выдачу по этим Part.id
    used = (
        db.session.query(IssuedPartRecord.id)
        .filter(IssuedPartRecord.part_id.in_(part_ids))
        .limit(1)
        .first()
    )

    # True => УЖЕ ЕСТЬ РАСХОД -> блокируем unpost
    return used is not None


# --- Receiving: post/unpost toggle -------------------------------------------

@inventory_bp.post("/receiving/<int:batch_id>/toggle", endpoint="receiving_toggle")
@login_required
def receiving_toggle(batch_id: int):
    from flask import flash, redirect, url_for, current_app
    from flask_login import current_user
    from extensions import db
    from models import ReceivingBatch
    from services.receiving import post_receiving_batch, unpost_receiving_batch

    # кто может жать Post / Unpost
    role_low = (getattr(current_user, "role", "") or "").strip().lower()
    is_adminish = (
        role_low in ("admin", "superadmin")
        or getattr(current_user, "is_admin", False)
        or getattr(current_user, "is_superadmin", False)
        or getattr(current_user, "is_super_admin", False)
    )
    if not is_adminish:
        flash("Access denied. Admin only.", "danger")
        return redirect(url_for("inventory.receiving_detail", batch_id=batch_id))

    batch = db.session.get(ReceivingBatch, batch_id)
    if not batch:
        flash("Batch not found.", "warning")
        return redirect(url_for("inventory.receiving_list"))

    status_now = (getattr(batch, "status", "") or "").strip().lower()

    try:
        if status_now != "posted":
            # DRAFT -> POST
            post_receiving_batch(
                batch.id,
                getattr(current_user, "id", None)
            )
            flash(
                f"Batch #{batch.id} posted and stock updated.",
                "success"
            )

        else:
            # POSTED -> UNPOST
            # сначала проверка - было ли уже списание этим деталям?
            if _batch_consumed_forbid_unpost(batch):
                # уже выдавали техникам -> запрещаем откат
                current_app.logger.debug(
                    "[RECEIVING_TOGGLE] Blocked UNPOST for batch %s: already consumed.",
                    batch.id,
                )
                flash(
                    "Cannot unpost: items from this batch were already issued to technicians.",
                    "danger"
                )
                return redirect(url_for("inventory.receiving_detail", batch_id=batch.id))

            # иначе можно безопасно откатить
            unpost_receiving_batch(
                batch.id,
                getattr(current_user, "id", None)
            )
            flash(
                f"Batch #{batch.id} reverted to draft and stock rolled back.",
                "warning"
            )

    except Exception as e:
        db.session.rollback()
        flash(f"Failed to toggle: {e}", "danger")

    return redirect(url_for("inventory.receiving_detail", batch_id=batch.id))

@inventory_bp.get("/receiving/new", endpoint="receiving_new")
@login_required
def receiving_new():
    from flask import flash, redirect, url_for
    from flask_login import current_user

    role = (getattr(current_user, "role", "") or "").lower()

    # Разрешаем admin и superadmin заходить на страницу создания НОВОГО батча
    if role not in ("admin", "superadmin"):
        flash("Access denied. only admin or superadmin can create Receiving batch.", "danger")
        return redirect(url_for("inventory.receiving_list"))

    # НИЧЕГО не флэшить про "only superadmin can edit Receiving"
    # потому что это не редактирование существующего, это НОВЫЙ (batch=None)

    return render_template(
        "receiving_edit.html",
        batch=None,
        today=datetime.utcnow().date()
    )


@inventory_bp.post("/receiving/<int:batch_id>/unpost", endpoint="receiving_unpost")
@login_required
def receiving_unpost(batch_id: int):
    if not _is_superadmin():
        flash("Access denied: only superadmin can unpost.", "danger")
        return redirect(url_for("inventory.receiving_detail", batch_id=batch_id))

    try:
        unpost_receiving_batch(batch_id, getattr(current_user, "id", None))
        flash("Batch unposted & stock rolled back.", "success")
    except Exception as e:
        current_app.logger.exception("Unpost failed")
        flash(f"Unpost failed: {e}", "danger")
    return redirect(url_for("inventory.receiving_detail", batch_id=batch_id))

@inventory_bp.post("/receiving/save", endpoint="receiving_save")
@login_required
def receiving_save():
    from flask import request, redirect, url_for, flash
    from datetime import datetime
    from flask_login import current_user

    from extensions import db
    from models import ReceivingBatch, ReceivingItem, Part
    from sqlalchemy import func

    f = request.form
    batch_id = (f.get("batch_id") or "").strip()
    action = (f.get("action") or "").strip().lower()
    force_existing = (f.get("force_existing") or "").strip() == "1"

    # ---------------- ROLE HELPERS ----------------
    role_low = (getattr(current_user, "role", "") or "").strip().lower()

    def is_super(u) -> bool:
        rl = (getattr(u, "role", "") or "").strip().lower()
        if rl == "superadmin":
            return True
        if getattr(u, "is_superadmin", False):
            return True
        if getattr(u, "is_super_admin", False):
            return True
        return False

    def is_adminish(u) -> bool:
        rl = (getattr(u, "role", "") or "").strip().lower()
        if rl in ("admin", "superadmin"):
            return True
        if getattr(u, "is_admin", False):
            return True
        if getattr(u, "is_superadmin", False):
            return True
        if getattr(u, "is_super_admin", False):
            return True
        return False

    user_is_super = is_super(current_user)
    user_is_adminish = is_adminish(current_user)

    # figure out: new batch or editing existing
    is_new_batch = (not batch_id) and (not force_existing)

    # ------------- ACCESS CONTROL -------------
    # правила:
    # 1) новый батч может создать admin / superadmin
    # 2) существующий батч:
    #    - draft: admin или superadmin может редактировать
    #    - posted: никто не редактирует тут (нужно сначала Unpost из toggle)
    if is_new_batch:
        if not user_is_adminish:
            flash("Access denied. Only admin or superadmin can create receiving.", "danger")
            return redirect(url_for("inventory.receiving_list"))
        batch = ReceivingBatch(created_by=getattr(current_user, "id", None))
        # заполним базовые поля так, чтобы не упасть на NOT NULL
        batch.status = "draft"
        batch.created_at = datetime.utcnow()
        # поставим supplier хотя бы пустую строку, но не None
        batch.supplier_name = (f.get("supplier_name") or "").strip() or "UNKNOWN"
        batch.invoice_number = (f.get("invoice_number") or "").strip() or None
        inv_date_raw = (f.get("invoice_date") or "").strip()
        inv_date_val = None
        if inv_date_raw:
            try:
                inv_date_val = datetime.strptime(inv_date_raw, "%Y-%m-%d").date()
            except Exception:
                try:
                    inv_date_val = datetime.strptime(inv_date_raw, "%m/%d/%Y").date()
                except Exception:
                    inv_date_val = None
        batch.invoice_date = inv_date_val
        batch.currency = (f.get("currency") or "USD").strip()[:8] or "USD"
        batch.notes = (f.get("notes") or "").strip() or None

    else:
        # existing batch
        try:
            batch = ReceivingBatch.query.get(int(batch_id))
        except Exception:
            batch = None

        if not batch:
            flash("Receiving batch not found.", "danger")
            return redirect(url_for("inventory.receiving_list"))

        # if already posted -> block edit
        status_low = (getattr(batch, "status", "") or "").strip().lower()
        if status_low == "posted":
            flash("Batch is posted. Unpost first, then edit.", "danger")
            return redirect(url_for("inventory.receiving_detail", batch_id=batch.id))

        # draft batch can be edited by admin/superadmin
        if not user_is_adminish:
            flash("Access denied. Only admin or superadmin can edit a draft batch.", "danger")
            return redirect(url_for("inventory.receiving_detail", batch_id=batch.id))

        # update header fields from form
        batch.supplier_name  = (f.get("supplier_name") or "").strip() or (batch.supplier_name or "UNKNOWN")
        batch.invoice_number = (f.get("invoice_number") or "").strip() or None

        inv_date_raw = (f.get("invoice_date") or "").strip()
        inv_date_val = None
        if inv_date_raw:
            try:
                inv_date_val = datetime.strptime(inv_date_raw, "%Y-%m-%d").date()
            except Exception:
                try:
                    inv_date_val = datetime.strptime(inv_date_raw, "%m/%d/%Y").date()
                except Exception:
                    inv_date_val = None
        batch.invoice_date = inv_date_val

        batch.currency = ((f.get("currency") or "USD").strip()[:8] or "USD")
        batch.notes    = (f.get("notes") or "").strip() or None

        # now rebuild lines
        batch.items.clear()

    # ---------- rebuild/add lines ----------
    idx = 0
    while True:
        basekey = f"rows[{idx}]"
        if not any(k.startswith(basekey) for k in f.keys()):
            break

        pn_val = (f.get(f"{basekey}[part_number]") or "").strip().upper()
        if pn_val:
            try:
                qty_val = int(f.get(f"{basekey}[quantity]") or 0)
            except:
                qty_val = 0
            try:
                cost_val = float(f.get(f"{basekey}[unit_cost]") or 0.0)
            except:
                cost_val = 0.0

            item = ReceivingItem(
                part_number = pn_val,
                part_name   = (f.get(f"{basekey}[part_name]") or "").strip(),
                quantity    = qty_val,
                unit_cost   = cost_val,
                location    = (f.get(f"{basekey}[location]") or "").strip()[:64],
            )
            batch.items.append(item)

        idx += 1

    # sync part names into Part table (non-destructive)
    if batch.items:
        for it in batch.items:
            pn_sync = (it.part_number or "").strip().upper()
            new_name = (it.part_name or "").strip()
            if not pn_sync or not new_name:
                continue
            part_obj = Part.query.filter(func.upper(Part.part_number) == pn_sync).first()
            if part_obj:
                old_name = (getattr(part_obj, "name", "") or "").strip()
                if old_name != new_name and new_name:
                    part_obj.name = new_name

    # save draft
    db.session.add(batch)
    db.session.commit()

    # if they clicked the green button "Post & Update Stock"
    if action == "post":
        # use your existing stock-posting service
        from services.receiving import post_receiving_batch
        post_receiving_batch(batch.id, getattr(current_user, "id", None))
        flash("Batch posted & stock updated.", "success")
    else:
        flash("Draft saved.", "success")

    return redirect(url_for("inventory.receiving_detail", batch_id=batch.id))

def _batch_has_been_consumed(batch):
    """
    True = из этого прихода уже что-то УШЛО техникам после того, как мы его провели.
    Значит:
      - нельзя менять количества строк
      - нельзя делать UNPOST
    False = приход ещё девственно чистый, его можно откатить/править.
    """
    # Если батч вообще не posted -> значит он ещё не попал на склад
    status_low = (getattr(batch, "status", "") or "").strip().lower()
    if status_low != "posted":
        return False

    posted_at = getattr(batch, "posted_at", None)
    if not posted_at:
        # теоретически не должно быть, но на всякий случай
        return False

    # Возьмём строки из batch (lines или items)
    lines = (getattr(batch, "lines", None)
             or getattr(batch, "items", None)
             or [])

    # Соберём part_numbers из строк прихода
    part_numbers = []
    for it in lines:
        pn = (getattr(it, "part_number", "") or "").strip()
        if pn:
            part_numbers.append(pn.upper())

    if not part_numbers:
        return False  # нечего проверять

    # Теперь найдём все Part.id, у которых Part.part_number in part_numbers
    # (сравниваем в верхнем регистре, как ты делаешь в других местах)
    # ВНИМАНИЕ: func.upper(Part.part_number).in_(...) не всегда красиво оптимизировано SQLite,
    # но для нас это ок.
    parts_q = (
        db.session.query(Part.id)
        .filter(func.upper(Part.part_number).in_(part_numbers))
        .all()
    )
    part_ids = [row[0] for row in parts_q]

    if not part_ids:
        return False  # строки прихода какие-то странные, но ок, считаем не потребляли

    # Проверяем: есть ли запись выдачи IssuedPartRecord по ЛЮБОМУ из этих part_id
    # с issue_date >= posted_at  (issue_date = когда отдали технику)
    used = (
        db.session.query(IssuedPartRecord.id)
        .filter(IssuedPartRecord.part_id.in_(part_ids))
        .filter(IssuedPartRecord.issue_date >= posted_at)
        .limit(1)
        .first()
    )

    return used is not None

@inventory_bp.get("/receiving/<int:batch_id>", endpoint="receiving_detail")
@login_required
def receiving_detail(batch_id):
    from models import ReceivingBatch, Part, IssuedPartRecord
    from sqlalchemy import func
    from flask import current_app

    log = current_app.logger

    batch = db.session.get(ReceivingBatch, batch_id)
    if not batch:
        flash("Batch not found.", "warning")
        return redirect(url_for("inventory.receiving_list"))

    # собрать строки батча
    lines = (
        getattr(batch, "lines", None)
        or getattr(batch, "items", None)
        or []
    )

    # ---------- helpers to compute money safely ----------
    def _to_int(x):
        try:
            return int(x)
        except Exception:
            try:
                return int(float(str(x).replace(",", ".")))
            except Exception:
                return 0

    def _to_float(x):
        try:
            return float(x)
        except Exception:
            try:
                return float(str(x).replace(",", ".").replace("$", ""))
            except Exception:
                return 0.0

    batch_total = 0.0
    for it in lines:
        qty = _to_int(getattr(it, "quantity", None) or getattr(it, "qty", 0))
        cost = _to_float(
            getattr(it, "unit_cost", None) or getattr(it, "unit_cost_at_issue", 0.0)
        )
        batch_total += qty * cost

    # ---------- detect "consumed" ----------
    def _batch_has_been_consumed(_batch):
        """
        consumed = True если мы видим, что хотя бы одна из деталей из этой партии
        уже фигурирует в IssuedPartRecord (то есть была выдана технику).

        Логика:
        1. Собираем все part_id напрямую из строк партии (если там есть поле part_id).
        2. Собираем все part_number из строк партии, находим им Part.id.
        3. Берём объединение этих Part.id.
        4. Проверяем IssuedPartRecord по этим Part.id.
        """

        status_low_local = (getattr(_batch, "status", "") or "").strip().lower()
        if status_low_local != "posted":
            # не posted => не считаем как "использовано", ещё черновик
            log.warning(f"[RECV_DETAIL consume-check] batch {batch_id}: not posted -> consumed=False")
            print(f"[RECV_DETAIL consume-check] batch {batch_id}: not posted -> consumed=False")
            return False

        # (1) part_ids напрямую из строк, если есть
        direct_part_ids = []
        for row in (
            getattr(_batch, "lines", None)
            or getattr(_batch, "items", None)
            or []
        ):
            if hasattr(row, "part_id") and getattr(row, "part_id") is not None:
                try:
                    direct_part_ids.append(int(row.part_id))
                except Exception:
                    pass

        # (2) part_numbers -> Part.id
        raw_pns = []
        for row in (
            getattr(_batch, "lines", None)
            or getattr(_batch, "items", None)
            or []
        ):
            pn_val = getattr(row, "part_number", None)
            if pn_val:
                pn_norm = str(pn_val).strip().upper()
                if pn_norm:
                    raw_pns.append(pn_norm)

        # делаем уникальные
        raw_pns = list({p for p in raw_pns})
        direct_part_ids = list({pid for pid in direct_part_ids})

        log.warning(f"[RECV_DETAIL consume-check] batch {batch_id}: raw_pns={raw_pns}, direct_part_ids(pre)={direct_part_ids}")
        print(f"[RECV_DETAIL consume-check] batch {batch_id}: raw_pns={raw_pns}, direct_part_ids(pre)={direct_part_ids}")

        # (2b) достаём ID по PN
        part_ids_from_pn = []
        if raw_pns:
            rows_parts = (
                db.session.query(Part.id, Part.part_number)
                .filter(func.upper(func.trim(Part.part_number)).in_(raw_pns))
                .all()
            )
            part_ids_from_pn = [row[0] for row in rows_parts]

            log.warning(f"[RECV_DETAIL consume-check] batch {batch_id}: part_ids_from_pn={part_ids_from_pn}, parts_matched={[p for _, p in rows_parts]}")
            print(f"[RECV_DETAIL consume-check] batch {batch_id}: part_ids_from_pn={part_ids_from_pn}")

        # (3) объединяем
        all_part_ids = set(direct_part_ids) | set(part_ids_from_pn)
        all_part_ids = list(all_part_ids)

        log.warning(f"[RECV_DETAIL consume-check] batch {batch_id}: all_part_ids(final)={all_part_ids}")
        print(f"[RECV_DETAIL consume-check] batch {batch_id}: all_part_ids(final)={all_part_ids}")

        if not all_part_ids:
            log.warning(f"[RECV_DETAIL consume-check] batch {batch_id}: no part ids -> consumed=False")
            print(f"[RECV_DETAIL consume-check] batch {batch_id}: no part ids -> consumed=False")
            return False

        # (4) проверяем, есть ли выдачи по этим Part.id
        issued_rows = (
            db.session.query(
                IssuedPartRecord.id,
                IssuedPartRecord.part_id,
                IssuedPartRecord.quantity,
                IssuedPartRecord.issue_date
            )
            .filter(IssuedPartRecord.part_id.in_(all_part_ids))
            .limit(5)
            .all()
        )

        log.warning(f"[RECV_DETAIL consume-check] batch {batch_id}: issued_rows={issued_rows}")
        print(f"[RECV_DETAIL consume-check] batch {batch_id}: issued_rows={issued_rows}")

        return len(issued_rows) > 0

    consumed = _batch_has_been_consumed(batch)

    # ---------- role / permissions ----------
    role_low = (getattr(current_user, "role", "") or "").lower()
    is_super = (
        role_low in ["admin", "superadmin"]
        or getattr(current_user, "is_admin", False)
        or getattr(current_user, "is_superadmin", False)
        or getattr(current_user, "is_super_admin", False)
    )

    status_low = (getattr(batch, "status", "") or "").strip().lower()
    is_posted = (status_low == "posted")

    # can_edit: супер + не consumed
    can_edit = bool(is_super and (not consumed))

    # can_unpost: супер + posted + не consumed
    can_unpost = bool(is_super and is_posted and (not consumed))

    log.warning(
        f"[RECV_DETAIL FLAGS] batch {batch_id} "
        f"status={batch.status} posted?={is_posted} consumed?={consumed} "
        f"is_super={is_super} => can_edit={can_edit} can_unpost={can_unpost}"
    )
    print(
        f"[RECV_DETAIL FLAGS] batch {batch_id} "
        f"status={batch.status} posted?={is_posted} consumed?={consumed} "
        f"is_super={is_super} => can_edit={can_edit} can_unpost={can_unpost}"
    )

    return render_template(
        "receiving_detail.html",
        batch=batch,
        lines=lines,
        batch_total=batch_total,

        readonly=(not can_edit),
        can_edit=can_edit,
        can_unpost=can_unpost,
        consumed=consumed,
    )

@inventory_bp.post("/receiving/<int:batch_id>/post", endpoint="receiving_post")
@login_required
def receiving_post(batch_id):
    from extensions import db
    from models import ReceivingBatch

    batch = db.session.get(ReceivingBatch, batch_id)
    if not batch:
        flash("Batch not found", "danger")
        return redirect(url_for("inventory.receiving_list"))

    status_now = (batch.status or "").strip().lower()

    # КРИТИЧЕСКИЙ ФИЛЬТР:
    # если партию уже когда-то фактически оприходовали (твой "призрак"),
    # то повторно склад трогать нельзя.
    #
    # Простая защита: если status уже 'posted' -> не зовём post_receiving_batch
    # (это уже было), НО также если status == 'draft', НО batch.posted_at уже не пустой,
    # значит её уже проводили и потом кто-то откатил только поле status.
    #
    already_posted_once = False
    if status_now == "posted":
        already_posted_once = True
    else:
        # иногда баг откатывает только поле status, но оставляет posted_at
        if getattr(batch, "posted_at", None):
            already_posted_once = True

    if already_posted_once:
        flash("Batch already applied to stock. Status updated to POSTED.", "info")
        # просто выставим статус в БД, без повторного прихода
        batch.status = "posted"
        db.session.commit()
        return redirect(url_for("inventory.receiving_detail", batch_id=batch_id))

    # нормальный путь: реально ещё не приходили -> делаем приход
    post_receiving_batch(batch_id, getattr(current_user, "id", None))
    flash("Batch posted", "success")
    return redirect(url_for("inventory.receiving_detail", batch_id=batch_id))

# __________________________________________________________________________________________________________
def create_receiving_from_rows(
    *,
    supplier_name: str,
    invoice_number: str | None,
    invoice_date: date | None,
    currency: str | None,
    notes: str | None,
    rows: list[dict],
    created_by=None,
    auto_post: bool = False
) -> ReceivingBatch:

    """
    Делает так:
      1. создаёт ReceivingBatch как draft + строки
      2. commit (чтоб был id)
      3. если auto_post == True:
            - вызывает post_receiving_batch(batch.id)
              (это плюсанёт склад и поставит status='posted')
            - перечитывает batch из базы (уже posted)
            - expunge() эту свежую версию, чтобы статус 'posted' не затёрся обратно
         и возвращает именно ЭТУ версию
      4. если auto_post == False:
            - просто возвращает draft, тоже expunge, чтоб не затирал потом
    """

    # 1. СОЗДАТЬ batch как draft
    batch = ReceivingBatch(
        supplier_name=(supplier_name or "").strip(),
        invoice_number=(invoice_number or "").strip() or None,
        invoice_date=invoice_date,
        currency=(currency or "USD").strip()[:8],
        notes=(notes or "").strip() or None,
        status="draft",
        created_at=datetime.utcnow(),
        created_by=created_by,
    )
    db.session.add(batch)
    db.session.flush()  # batch.id теперь есть

    # 1.5. СОЗДАТЬ строки
    line_no = 1
    for r in rows:
        pn = (r.get("part_number") or r.get("pn") or "").strip()
        if not pn:
            continue

        qty = int((r.get("quantity") or r.get("qty") or 0) or 0)
        if qty <= 0:
            continue

        line_kwargs = dict(
            line_no=line_no,
            part_number=pn,
            part_name=(r.get("part_name") or r.get("description") or r.get("descr") or "").strip() or None,
            quantity=qty,
            unit_cost=float((r.get("unit_cost") or r.get("price") or 0) or 0),
            location=(r.get("location") or r.get("supplier") or "").strip() or None,
        )

        # ВАЖНО: у тебя ReceivingItem сейчас создаётся с batch_id=gr.id
        # если у твоей модели строк реально поле называется goods_receipt_id,
        # то замени здесь на goods_receipt_id=batch.id
        line_kwargs["batch_id"] = batch.id

        line = ReceivingItem(**line_kwargs)
        db.session.add(line)

        line_no += 1

    # 2. commit draft+строки, теперь batch и items в базе
    db.session.commit()

    # 3. если НЕ нужно автопостить → просто вернуть draft (но вынести из сессии)
    if not auto_post:
        fresh_draft = db.session.get(ReceivingBatch, batch.id)
        try:
            from flask import current_app
            current_app.logger.debug(
                "### DEBUG create_receiving_from_rows draft return id=%s status=%s",
                fresh_draft.id,
                fresh_draft.status,
            )
        except Exception:
            pass

        db.session.expunge(fresh_draft)
        return fresh_draft

    # 4. auto_post == True → проводим на склад и ставим posted
    post_receiving_batch(batch.id, current_user_id=created_by)

    # перечитываем уже ПОСТНУТУЮ версию
    posted_batch = db.session.get(ReceivingBatch, batch.id)

    try:
        from flask import current_app
        current_app.logger.debug(
            "### DEBUG create_receiving_from_rows AFTER POST id=%s status=%s",
            posted_batch.id,
            posted_batch.status,
        )
    except Exception:
        pass

    # КРИТИЧЕСКИЙ МОМЕНТ:
    # вынимаем объект из сессии, чтобы никакой следующий commit в этом же request
    # не смог случайно залить обратно старый статус 'draft'
    db.session.expunge(posted_batch)

    try:
        from flask import current_app
        current_app.logger.debug(
            "### DEBUG CFR (create_receiving_from_rows) RETURNING batch_id=%s status=%s auto_post=%s file=%s",
            gr.id,
            getattr(gr, "status", None),
            auto_post,
            __file__,
        )
    except Exception:
        pass

    return posted_batch


def _norm_cols(df):
    """
    Нормализация колонок DF к именам: part_number, part_name, quantity, unit_cost, supplier, location.
    + Фикс: разлепить 'PN DESCRIPTION' -> PN и DESCRIPTION по первому пробелу.
    + Отсев: 'Shipment marking/Shipmentmarking' и итоговые строки (Order total/Subtotal/...).
    """
    import re
    cols = {c.lower().strip(): c for c in df.columns}

    def pick(*names):
        for n in names:
            for key, orig in cols.items():
                if key == n.lower():
                    return orig
        # «умные» совпадения
        for key, orig in cols.items():
            if any(k in key for k in names):
                return orig
        return None

    c_pn  = pick("PART #","PART#","PART","PN","part_number")
    c_nm  = pick("DESCR.","DESCRIPTION","DESC","NAME","part_name")
    c_qty = pick("QTY","QUANTITY","qty ordered","qty shipped","quantity")
    c_uc  = pick("UNIT COST","UNIT PRICE","COST","PRICE","unit_cost")
    c_sup = pick("SUPPLIER","VENDOR","FROM","supplier")
    c_loc = pick("LOCATION","BIN","SHELF","location")

    # --- Регэкспы и фильтры ---
    marker_rx = re.compile(r'^\s*shipment\s*mark(?:ing)?\s*[:\-]?', re.I)
    totals_rx = re.compile(r'^\s*(order\s*total|orderinetot|ordertot|subtotal|total)\b', re.I)
    # допустимый токен PN: буквы/цифры и часто используемые символы ('-','/','.', '_')
    pn_token_rx = re.compile(r'^[A-Z0-9][A-Z0-9\-\/\._]*$', re.I)

    def split_pn_desc(text: str):
        """Разделить 'PN rest...' -> (PN, 'rest...') если первый токен похож на PN."""
        s = (text or "").strip()
        if not s:
            return "", ""
        parts = s.split()
        if len(parts) <= 1:
            return s, ""
        first, rest = parts[0], " ".join(parts[1:])
        if pn_token_rx.match(first):
            return first, rest
        return "", s  # первый токен не похож на PN — считаем всё описанием

    out = []
    for _, row in df.iterrows():
        raw_pn = str(row.get(c_pn, "")).strip() if c_pn else ""
        raw_nm = str(row.get(c_nm, "")).strip() if c_nm else ""

        # 1) Отсев мусора: Shipment marking, Order total и т.п.
        if (raw_pn and marker_rx.match(raw_pn)) or (raw_nm and marker_rx.match(raw_nm)):
            continue
        if totals_rx.match(raw_pn) or totals_rx.match(raw_nm):
            continue

        # 2) Приведение PN/NAME
        pn, nm = raw_pn, raw_nm

        # a) случай: PN склеен с описанием в самом PN (есть пробел)
        if pn and " " in pn:
            p_guess, rest = split_pn_desc(pn)
            if p_guess:
                pn = p_guess
                # если описания не было — берём остаток; если было — приписываем справа
                nm = (nm or "").strip()
                nm = (rest if not nm else f"{nm} {rest}").strip()

        # b) случай: PN пуст, а в DESCRIPTION лежит "PN rest..."
        if not pn and nm:
            p_guess, rest = split_pn_desc(nm)
            if p_guess:
                pn = p_guess
                nm = rest

        # c) иногда обе колонки одинаковые (дубликат) — очистим описание
        if nm and pn and nm.strip().upper() == pn.strip().upper():
            nm = ""

        # 3) Числа
        qty_raw = str(row.get(c_qty, "")).replace(",", "") if c_qty else ""
        uc_raw  = str(row.get(c_uc, "")).replace("$", "").replace(",", "") if c_uc else ""
        try:
            qty = int(float(qty_raw or "0"))
        except Exception:
            qty = 0
        try:
            uc = float(uc_raw or "0")
        except Exception:
            uc = 0.0

        # 4) Пустые строки выкидываем
        if not pn and not nm and qty == 0 and uc == 0.0:
            continue

        out.append({
            "part_number": pn,
            "part_name": nm,
            "quantity": qty,
            "unit_cost": uc,
            "supplier": (str(row.get(c_sup, "")).strip() if c_sup else ""),
            "location": (str(row.get(c_loc, "")).strip() if c_loc else ""),
        })
    return out

@inventory_bp.get("/receiving/import")
@login_required
def receiving_import_form():
    """Форма загрузки PDF (или CSV, если захочешь)."""
    return render_template("receiving_import_upload.html")

@inventory_bp.route("/receiving/import", methods=["GET", "POST"])
@login_required
def receiving_import_upload():
    # legacy path disabled, use /import-parts instead
    flash("Use Import Parts page instead.", "warning")
    return redirect(url_for("inventory.import_parts_upload"))

@inventory_bp.get("/receiving/<int:batch_id>/attachment")
@login_required
def download_receiving_attachment(batch_id):
    import os
    from flask import send_file, abort
    from models import GoodsReceipt

    gr = GoodsReceipt.query.get_or_404(batch_id)
    p = (gr.attachment_path or "").strip()
    if not p or not os.path.exists(p):
        abort(404)
    return send_file(
        p,
        mimetype="application/pdf",
        as_attachment=False,
        download_name=os.path.basename(p)
    )

@inventory_bp.get("/add-batch", endpoint="add_part_batch_form")
@login_required
def add_part_batch_form():
    """
    Scan / Add Multiple Parts UI.
    Показываем страницу с полем сканера и таблицей batchItems (JS).
    """
    # доступы можно ограничить если хочешь:
    # if (current_user.role or '').lower() not in ('admin','superadmin'):
    #     flash("Access denied", "danger")
    #     return redirect(url_for("inventory.dashboard"))

    return render_template("add_part.html")

@inventory_bp.post("/add-batch", endpoint="add_part_batch")
@login_required
def add_part_batch():
    """
    Новый поток:
    1. Читаем batch_payload (массив деталей).
    2. Создаём ReceivingBatch (черновик) -> это GoodsReceipt.
    3. Создаём ReceivingItem строки -> это GoodsReceiptLine.
    4. Коммитим draft, чтобы получить batch.id.
    5. Делаем post_receiving_batch(batch.id) → это:
          - прибавит qty к складу ОДИН РАЗ
          - выставит batch.status = 'posted'
    6. Редиректим на receiving_detail(batch_id).
    """

    role_low = (getattr(current_user, "role", "") or "").lower()
    if role_low not in ("admin", "superadmin"):
        flash("Access denied", "danger")
        return redirect(url_for("inventory.dashboard"))

    payload_raw = request.form.get("batch_payload", "[]").strip()
    try:
        items = json.loads(payload_raw)
    except Exception:
        current_app.logger.exception("Bad batch_payload JSON")
        flash("Invalid batch data", "danger")
        return redirect(url_for("inventory.dashboard"))

    # Пример items:
    # [
    #   {
    #     "part_number": "WR14X27232",
    #     "name": "GASKET DOOR FF WAV",
    #     "qty": 2,
    #     "cost": "37.98",
    #     "loc": "A2/B2",
    #     "stock_now": "5"
    #   },
    #   ...
    # ]

    # ---------- 1. создаём сам batch в status='draft' ----------
    supplier_hint = "ADD-BATCH"
    invoice_hint  = datetime.utcnow().strftime("BULK-%Y%m%d-%H%M%S")

    batch = ReceivingBatch(
        supplier_name = supplier_hint,
        invoice_number = invoice_hint,
        invoice_date = datetime.utcnow().date(),
        currency = "USD",
        notes = "Created via /add-batch",
        status = "draft",
        created_at = datetime.utcnow(),
        created_by = getattr(current_user, "id", None),
    )
    db.session.add(batch)
    db.session.flush()   # теперь batch.id существует, но ещё не commit

    # ---------- 2. добавляем строки ----------
    line_no = 1
    for row in items:
        pn  = (row.get("part_number") or "").strip().upper()
        nm  = (row.get("name") or "").strip()
        loc = (row.get("loc") or "").strip().upper()

        # qty
        try:
            qty = int(row.get("qty") or 0)
        except Exception:
            qty = 0

        # Пропускаем пустые/некорректные ряды
        if qty <= 0 or not pn:
            continue

        # cost
        try:
            unit_cost_val = float(row.get("cost")) if row.get("cost") not in (None, "") else 0.0
        except Exception:
            unit_cost_val = 0.0

        # ВАЖНО: в новой модели FK называется goods_receipt_id, не batch_id.
        line = ReceivingItem(
            goods_receipt_id = batch.id,
            line_no          = line_no,
            part_number      = pn,
            part_name        = nm or None,
            quantity         = qty,
            unit_cost        = unit_cost_val,
            location         = loc or None,
        )
        db.session.add(line)
        line_no += 1

    # ---------- 3. коммитим draft+строки ----------
    # Сейчас batch.status всё ещё "draft", ничего не прибавлено в склад.
    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        current_app.logger.exception("Failed to commit draft receiving batch")
        flash(f"Failed to save new batch: {e}", "danger")
        return redirect(url_for("inventory.dashboard"))

    # ---------- 4. официально постим через сервис ----------
    # post_receiving_batch(batch.id, user_id) должен:
    #   - прибавить qty в Part.quantity/on_hand
    #   - переключить status -> 'posted'
    from services.receiving import post_receiving_batch
    try:
        post_receiving_batch(batch.id, getattr(current_user, "id", None))
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        current_app.logger.exception("Failed to post receiving batch")
        flash(f"Batch saved as draft but failed to post stock: {e}", "warning")
        return redirect(url_for("inventory.receiving_detail", batch_id=batch.id))

    flash(f"Stock received and posted. Batch #{batch.id}", "success")
    return redirect(url_for("inventory.receiving_detail", batch_id=batch.id))


















