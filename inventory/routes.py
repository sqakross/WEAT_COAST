from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import func, or_, and_
from models import IssuedPartRecord, WorkOrder, WorkOrderPart, TechReceiveLog, IssuedBatch, Part
import json
# from extensions import db
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
from datetime import datetime
from collections import defaultdict

from config import Config
from extensions import db
# from models import Part, IssuedPartRecord, User
from utils.invoice_generator import generate_invoice_pdf
# from models.order_items import OrderItem
from models import User, ROLE_SUPERADMIN, ROLE_ADMIN, ROLE_USER, ROLE_VIEWER,Part, WorkOrder, WorkOrderPart
from sqlalchemy import or_
from pathlib import Path

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

_PREFIX_RE = re.compile(r'^\s*(wo|job|pn|model|brand|serial)\s*:\s*(.+)\s*$', re.I)

_units_re = re.compile(r"^units\[(\d+)\]\[(brand|model|serial)\]$")
_rows_re  = re.compile(r"^units\[(\d+)\]\[rows\]\[(\d+)\]\[(part_number|part_name|quantity|alt_numbers|supplier|backorder_flag|line_status|unit_cost)\]$")

# --- Invoice numbering baseline ---
INVOICE_START_AT = 140  # новые инвойсы начнутся с 000140

# inventory/routes.py  (добавь рядом с _create_batch_for_records)

def _is_return_row(r) -> bool:
    """Возвратная строка — это строка с отрицательным количеством."""
    try:
        return (r.quantity or 0) < 0
    except Exception:
        return False

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
            current_app.logger.warning(msg)
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

@inventory_bp.get("/work_orders/new", endpoint="wo_new")
@login_required
def wo_new():
    # только admin/superadmin
    role = (getattr(current_user, "role", "") or "").strip().lower()
    if role not in ("admin", "superadmin"):
        flash("Access denied", "danger")
        return redirect(url_for("inventory.wo_list"))

    # болванка: один appliance и пустая строка
    units = [{
        "brand": "", "model": "", "serial": "",
        "rows": [{
            "id": None,
            "part_number": "", "part_name": "", "quantity": 1,
            "alt_numbers": "",
            "warehouse": "",
            "supplier": "",
            "backorder_flag": False,
            "line_status": "search_ordered",
            "unit_cost": 0.0,
        }],
    }]

    recent_suppliers = session.get("recent_suppliers", []) or []

    # Важно: multi-форма
    return render_template(
        "wo_form_units.html",
        wo=None,
        units=units,
        recent_suppliers=recent_suppliers,
        readonly=False,
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
@inventory_bp.get("/work_orders/<int:wo_id>", endpoint="wo_detail")
@login_required
def wo_detail(wo_id):
    """
    Work Order details page (flat + multi-appliance).
    """
    from flask import current_app, render_template, request, flash, redirect, url_for
    from sqlalchemy import func
    from sqlalchemy.orm import selectinload
    from models import WorkOrder, WorkUnit, WorkOrderPart
    from extensions import db

    try:
        wo = (
            db.session.query(WorkOrder)
            .options(
                selectinload(WorkOrder.parts),
                selectinload(WorkOrder.units).selectinload(WorkUnit.parts),
            )
            .get(wo_id)
        )
    except Exception as e:
        current_app.logger.exception("Failed to load WorkOrder %s", wo_id)
        wo = None

    if not wo:
        flash(f"Work Order #{wo_id} not found.", "danger")
        return redirect(url_for("inventory.wo_list"))

    # suppliers
    suppliers = []
    try:
        rows = (
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
        suppliers = [r[0] for r in rows]
    except Exception:
        current_app.logger.exception("Suppliers query failed")
        suppliers = []

    # batches — оставляем пусто пока
    batches = []

    return render_template(
        "wo_detail.html",
        wo=wo,
        avail=[],
        batches=batches,
        suppliers=suppliers,
    )


@inventory_bp.post("/work_orders/new", endpoint="wo_create")
@login_required
def wo_create():
    if getattr(current_user, "role", "") not in ("admin", "superadmin"):
        flash("Access denied", "danger")
        return redirect(url_for("inventory.wo_list"))

    from models import WorkOrder, WorkOrderPart
    from extensions import db

    f = request.form
    technician_name = (f.get("technician_name") or "").strip()
    job_numbers = (f.get("job_numbers") or "").strip()
    brand = (f.get("brand") or "").strip()
    model = (f.get("model") or "").strip()
    serial = (f.get("serial") or "").strip()
    job_type = (f.get("job_type") or "BASE").strip().upper()

    def _f(x, default=0.0):
        try:
            return float(x)
        except Exception:
            return float(default)

    delivery_fee = _f(f.get("delivery_fee"), 0)
    markup_percent = _f(f.get("markup_percent"), 0)

    if not technician_name or not job_numbers:
        flash("Technician and Job(s) are required.", "danger")
        return redirect(url_for("inventory.wo_new"))

    # канонический job = максимальный из перечисленных чисел
    jobs = [j.strip() for j in job_numbers.replace(",", " ").split() if j.strip()]
    nums = []
    for j in jobs:
        try:
            nums.append(int(j))
        except Exception:
            pass
    # canonical_job у тебя свойство @property в модели — в БД не пишем
    # используем только job_numbers
    wo = WorkOrder(
        technician_name=technician_name,
        job_numbers=", ".join(jobs),
        brand=brand, model=model, serial=serial,
        job_type=job_type if job_type in ("BASE","INSURANCE") else "BASE",
        delivery_fee=delivery_fee,
        markup_percent=markup_percent,
        status="search_ordered",
    )
    db.session.add(wo)
    db.session.flush()

    # полезные клипперы
    def _clip20(s: str) -> str:
        return (s or "").strip()[:20]

    def _clip6(s: str) -> str:
        return (s or "").strip()[:6]

    # парсим строки parts из формы: rows[i][field]
    import re
    pat = re.compile(r"^rows\[(\d+)\]\[(\w+)\]$")
    tmp = {}
    for k, v in f.items():
        m = pat.match(k)
        if not m:
            continue
        i = int(m.group(1)); field = m.group(2)
        tmp.setdefault(i, {})[field] = (v or "").strip()

    for i in sorted(tmp.keys()):
        row = tmp[i]

        pn = _clip20((row.get("part_number") or "").upper())
        if not pn:
            continue

        # поддерживаем alt_numbers и alt_part_numbers
        alt_raw = (row.get("alt_numbers") or row.get("alt_part_numbers") or "")
        alt_tokens = [ _clip20(t) for t in alt_raw.split(",") if t is not None ]
        alt_joined = ",".join(alt_tokens)

        supplier = _clip6(row.get("supplier") or "")

        try:
            qty = int(row.get("quantity") or 0)
        except Exception:
            qty = 0

        part = WorkOrderPart(
            work_order_id=wo.id,
            part_number=pn,                 # ≤20
            alt_numbers=alt_joined,         # алиас к alt_part_numbers
            part_name=(row.get("part_name") or "").strip(),
            quantity=qty,
            supplier=supplier,              # ≤6
            backorder_flag=bool(row.get("backorder_flag")),
            status="search_ordered",
        )
        db.session.add(part)

    db.session.commit()
    flash("Work order created.", "success")
    return redirect(url_for("inventory.wo_detail", wo_id=wo.id))


# @inventory_bp.get("/work_orders/new")
# @login_required
# def wo_new():
#     if getattr(current_user, "role", "") not in ("admin", "superadmin"):
#         flash("Access denied", "danger")
#         return redirect(url_for("inventory.wo_list"))
#     # сразу используем форму с юнитами (как в wo_newx)
#     units = [{
#         "brand": "", "model": "", "serial": "",
#         "rows": [{"part_number": "", "part_name": "", "quantity": 1,
#                   "alt_numbers": "", "supplier": "", "backorder_flag": False, "line_status": "search_ordered"}]
#     }]
#     return render_template("wo_form_units.html", wo=None, units=units)

@inventory_bp.post("/work_orders/save", endpoint="wo_save")
@login_required
def wo_save():
    from flask import request, session, redirect, url_for, flash, current_app
    from models import WorkOrder, WorkUnit, WorkOrderPart
    from extensions import db
    import re

    # доступ
    if (getattr(current_user, "role", "") or "").lower() not in ("admin", "superadmin"):
        flash("Access denied", "danger")
        return redirect(url_for("inventory.wo_list"))

    f = request.form
    log_keys = sorted(list(f.keys()))
    current_app.logger.warning("WO_SAVE keys (%s): %s", len(log_keys), log_keys[:200])

    def _clip(s, n): return (s or "").strip()[:n]
    def _i(x, default=0):
        try: return int(x)
        except: return int(default)
    def _f(x, default=None):
        if x is None or x == "": return default
        try: return float(x)
        except: return default

    wo_id = (f.get("wo_id") or "").strip()
    is_new = not wo_id

    # создать/получить заказ
    if is_new:
        wo = WorkOrder()
        db.session.add(wo)
    else:
        wo = WorkOrder.query.get_or_404(int(wo_id))

    # шапка
    wo.technician_name = (f.get("technician_name") or "").strip().upper()
    wo.job_numbers     = (f.get("job_numbers") or "").strip()
    wo.job_type        = (f.get("job_type") or "BASE").strip().upper()
    wo.delivery_fee    = _f(f.get("delivery_fee"), 0) or 0.0
    wo.markup_percent  = _f(f.get("markup_percent"), 0) or 0.0

    st = (f.get("status") or "search_ordered").strip()
    wo.status = st if st in ("search_ordered", "ordered", "done") else "search_ordered"

    # витринные хедеры (если присланы)
    brand_hdr  = (f.get("brand")  or "").strip()
    model_hdr  = _clip(f.get("model"), 25)
    serial_hdr = _clip(f.get("serial"), 25)
    if brand_hdr:  wo.brand  = brand_hdr
    if model_hdr:  wo.model  = model_hdr
    if serial_hdr: wo.serial = serial_hdr

    # ---- REGEX-парсер units/rows (включая ordered_flag) ----
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
            units_map[ui]["rows"][ri][name] = f.get(key)

    # собрать payload
    units_payload = []
    new_rows_count = 0
    for ui in sorted(units_map.keys()):
        u = units_map[ui]
        rows = []
        for ri in sorted(u.get("rows", {}).keys()):
            r = u["rows"][ri]
            pn  = (r.get("part_number") or "").strip().upper()
            qty = _i(r.get("quantity") or 0, 0)

            alt_raw  = (r.get("alt_numbers") or r.get("alt_part_numbers") or "").strip()
            wh_raw   = (r.get("warehouse")  or r.get("unit_label") or "").strip()
            sup_raw  = (r.get("supplier")   or r.get("supplier_name") or "").strip()
            ucost    = _f(r.get("unit_cost"), None)
            bo_flag  = bool(r.get("backorder_flag"))
            lstatus  = (r.get("status") or "search_ordered").strip()

            # hidden ordered_flag (0/1) — JS синхронизирует при submit
            try:
                ord_flag = bool(int(r.get("ordered_flag") or 0))
            except Exception:
                ord_flag = False

            if pn and qty > 0:
                new_rows_count += 1
                rows.append({
                    "part_number": _clip(pn, 80),
                    "part_name":   _clip(r.get("part_name"), 120),
                    "quantity":    qty,
                    "alt_numbers": _clip(alt_raw, 200),
                    "warehouse":   _clip(wh_raw, 120),
                    "supplier":    _clip(sup_raw, 80),
                    "unit_cost":   ucost,
                    "backorder_flag": bo_flag,
                    "line_status": lstatus if lstatus in ("search_ordered","ordered","done") else "search_ordered",
                    "ordered_flag": ord_flag,
                })

        brand  = (u.get("brand")  or "").strip()
        model  = _clip(u.get("model"), 25)
        serial = _clip(u.get("serial"), 25)
        if rows or any([brand, model, serial]):
            units_payload.append({"brand": brand, "model": model, "serial": serial, "rows": rows})

    current_app.logger.warning("WO_SAVE parsed: units=%s, rows=%s",
                               len(units_payload),
                               sum(len(x.get('rows') or []) for x in units_payload))

    # 1) Ничего не добавили/не изменили — сохраняем только шапку и остаёмся в detail
    if new_rows_count == 0:
        db.session.commit()
        flash("Work Order saved.", "success")
        return redirect(url_for("inventory.wo_detail", wo_id=wo.id))

    # 2) Если rows распарсились пустыми — не трогаем старые parts
    if not units_payload:
        flash("Nothing parsable in rows. Nothing was changed.", "warning")
        db.session.rollback()
        return redirect(url_for("inventory.wo_detail", wo_id=wo.id))

    # Удаляем прежние units/parts только теперь, когда уверены, что есть что записать
    for u in list(getattr(wo, "units", []) or []):
        for p in list(getattr(u, "parts", []) or []):
            db.session.delete(p)
        db.session.delete(u)
    db.session.flush()

    # витринные поля: если пустые — возьмём из первого юнита
    def _first_nonempty(lst, key):
        for it in lst or []:
            v = (it.get(key) or "").strip()
            if v: return v
        return ""
    if not (wo.brand or "").strip():
        wo.brand  = _clip(_first_nonempty(units_payload, "brand"), 40)
    if not (wo.model or "").strip():
        wo.model  = _clip(_first_nonempty(units_payload, "model"), 25)
    if not (wo.serial or "").strip():
        wo.serial = _clip(_first_nonempty(units_payload, "serial"), 25)

    suppliers_seen = []

    for up in units_payload:
        unit = WorkUnit(
            work_order=wo,
            brand=(up.get("brand") or "").strip(),
            model=_clip(up.get("model"), 25),
            serial=_clip(up.get("serial"), 25),
        )
        db.session.add(unit)
        db.session.flush()

        for r in (up.get("rows") or []):
            sup = r.get("supplier") or ""
            if sup:
                s_norm = " ".join(sup.split())
                if s_norm and s_norm.lower() not in [x.lower() for x in suppliers_seen]:
                    suppliers_seen.append(s_norm)

            # финальный ordered — учитываем статус WO
            ord_final = bool(r.get("ordered_flag")) if (wo.status == "ordered") else False

            wop = WorkOrderPart(
                work_order=wo,
                unit=unit,
                part_number=r["part_number"],
                part_name=(r.get("part_name") or None),
                quantity=int(r.get("quantity") or 0),
                alt_part_numbers=(r.get("alt_numbers") or None),
                supplier=(sup or None),
                backorder_flag=bool(r.get("backorder_flag")),
                status=("ordered" if ord_final else "search_ordered"),
            )

            if hasattr(wop, "warehouse"):
                wop.warehouse = (r.get("warehouse") or "")[:120]
            wop.unit_label = (r.get("warehouse") or "")[:120] or None

            if hasattr(wop, "unit_cost"):
                uc = r.get("unit_cost")
                if uc is not None and uc != "":
                    try: wop.unit_cost = float(uc)
                    except: wop.unit_cost = None

            if hasattr(wop, "ordered_flag"):
                wop.ordered_flag = ord_final

            db.session.add(wop)

    db.session.commit()

    # recent_suppliers
    if suppliers_seen:
        cur = session.get("recent_suppliers", [])
        merged, seen = [], set()
        for x in suppliers_seen + list(cur):
            xl = x.lower()
            if xl in seen: continue
            seen.add(xl)
            merged.append(x)
        session["recent_suppliers"] = merged[:20]
        session.modified = True

    flash("Work Order saved.", "success")
    # после сохранения — в detail (Work Order #N)
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
        q = q.distinct(WorkOrder.id)

    # Твой прежний порядок/лимит оставляем
    items = q.order_by(WorkOrder.created_at.desc()).limit(200).all()

    return render_template("wo_list.html", items=items, args=request.args)

@inventory_bp.post("/work_orders/<int:wo_id>/issue_instock", endpoint="wo_issue_instock")
@login_required
def wo_issue_instock(wo_id):
    if getattr(current_user, "role", "") not in ("admin", "superadmin"):
        flash("Access denied", "danger")
        return redirect(url_for("inventory.wo_detail", wo_id=wo_id))

    # базовые импорты, которые используются ниже
    from urllib.parse import urlencode
    from markupsafe import Markup
    from sqlalchemy import func
    from datetime import datetime
    from flask import request, session

    from extensions import db
    from models import WorkOrder, WorkOrderPart, Part  # Part — складская позиция

    wo = WorkOrder.query.get_or_404(wo_id)

    # ===== прочитаем флаг автосмены статуса =====
    set_status = (request.form.get("set_status") or "").strip().lower()

    # === 1) расчёт доступности (для greedy-проверки) ===
    try:
        avail_rows = compute_availability(wo) or []
    except Exception:
        avail_rows = []

    stock_map = {}  # PN -> on_hand
    hint_map  = {}  # PN -> hint
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

    # === 2) Режим “выбрано” — читаем реальные ID строк ===
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
            .filter(WorkOrderPart.work_order_id == wo_id,
                    WorkOrderPart.id.in_(ids))
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

            # Найдём складскую запись (location-специфично, если есть)
            q_base = Part.query.filter(func.upper(Part.part_number) == pn)
            part = None
            if part_has_location and getattr(line, "warehouse", None):
                part = q_base.filter(
                    func.coalesce(Part.location, "") == (line.warehouse or "")
                ).first()
            if not part:
                part = q_base.first()

            ok = can_issue(pn, qty_req)

            # Фоллбэк: если greedy «нет», но реально хватает — разрешим
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
                skipped_rows.append({
                    "id": line.id,
                    "pn": pn,
                    "name": getattr(line, "part_name", "") or "—",
                    "qty": qty_req,
                    "hint": hint_norm
                })
                continue

            unit_price = None
            if getattr(line, "unit_cost", None) is not None:
                try:
                    unit_price = float(line.unit_cost)
                except Exception:
                    unit_price = None

            items_to_issue.append({
                "part_id": part.id,
                "qty": qty_req,
                "unit_price": unit_price
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

        # Поддержка двух сигнатур: (dt, records) ИЛИ (dt, created_bool)
        issue_date = None
        new_records = None
        try:
            issue_date, new_records = issue_result  # новая сигнатура
        except Exception:
            issue_date, _created = issue_result     # старая сигнатура

        # отметим строки
        now = datetime.utcnow()
        for line in WorkOrderPart.query.filter(WorkOrderPart.id.in_(issued_row_ids)).all():
            if hasattr(line, "issued_qty"):
                line.issued_qty = (line.issued_qty or 0) + int(line.quantity or 0)
            if hasattr(line, "last_issued_at"):
                line.last_issued_at = now
        db.session.commit()

        # === ВАЖНО: меняем статус, если попросили ===
        if set_status == "done":
            try:
                wo.status = "done"
                db.session.commit()
            except Exception:
                db.session.rollback()

        # --- Если есть записи — сразу делаем инвойс и редиректим на него ---
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
                # упадёт — ниже дадим ссылку на grouped

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

        return redirect(url_for("inventory.wo_detail",
                                wo_id=wo.id,
                                issued_ids=",".join(map(str, issued_row_ids))))

    # === 3) «ничего не выбрано» — прежний поток ===
    items_to_issue.clear()
    for r in avail_rows:
        if int(r.get("issue_now") or 0) > 0:
            pn = (r.get("part_number") or "").strip().upper()
            if not pn:
                continue
            part = Part.query.filter(func.upper(Part.part_number) == pn).first()
            if not part:
                continue
            items_to_issue.append({
                "part_id": part.id,
                "qty": int(r["issue_now"]),
                "unit_price": None
            })

    if not items_to_issue:
        flash("Nothing available to issue (all WAIT).", "warning")
        return redirect(url_for("inventory.wo_detail", wo_id=wo_id))

    issue_date, _created_or_records = _issue_records_bulk(
        issued_to=wo.technician_name,
        reference_job=wo.canonical_job,
        items=items_to_issue
    )

    # если явно попросили — ставим done
    if set_status == "done":
        try:
            wo.status = "done"
            db.session.commit()
        except Exception:
            db.session.rollback()
    else:
        # иначе — по факту остатка
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

    # grouped отчёт
    d = (issue_date or datetime.utcnow()).date().isoformat()
    params = urlencode({
        "start_date": d,
        "end_date": d,
        "recipient": wo.technician_name,
        "reference_job": wo.canonical_job
    })
    link = f"/reports_grouped?{params}"
    flash(Markup(f'Issued in-stock items. <a href="{link}" target="_blank" rel="noopener">Open invoice group</a> to print.'), "success")
    return redirect(url_for("inventory.wo_detail", wo_id=wo.id))

@inventory_bp.get("/work_orders/newx")
@login_required
def wo_newx():
    from flask import session
    recent_suppliers = session.get("recent_suppliers", [])
    # только admin/superadmin
    if getattr(current_user, "role", "") not in ("admin", "superadmin"):
        flash("Access denied", "danger")
        return redirect(url_for("inventory.wo_list"))
    # Пустая форма с одним unit и одной строкой
    units = [{
        "brand": "", "model": "", "serial": "",
        "rows": [{"part_number": "", "part_name": "", "quantity": 1,
                  "alt_numbers": "", "supplier": "", "backorder_flag": False, "line_status": "search_ordered"}]
    }]
    return render_template(
        "wo_form.html",  # было "inventory/wo_form.html"
        wo=None,
        units=units,
        readonly=False,
        recent_suppliers=recent_suppliers
    )


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

    # Преобразуем ORM → payload для multi-шаблона
    units = []
    for u in (getattr(wo, "units", []) or []):
        rows = []
        for p in (getattr(u, "parts", []) or []):
            rows.append({
                "id":             getattr(p, "id", None),
                "part_number":    getattr(p, "part_number", "") or "",
                "part_name":      getattr(p, "part_name", "") or "",
                "quantity":       int(getattr(p, "quantity", 0) or 0),
                "alt_numbers":    getattr(p, "alt_numbers", "") or "",
                # соответствие твоему savex:
                "warehouse":      getattr(p, "warehouse", "") or "",
                "supplier":       getattr(p, "supplier", "") or "",
                "backorder_flag": bool(getattr(p, "backorder_flag", False)),
                "line_status":    getattr(p, "line_status", "") or "search_ordered",
                "unit_cost":      (
                    float(getattr(p, "unit_cost")) if getattr(p, "unit_cost") is not None else None
                ),
            })
        # если в юните нет строк — добавим одну пустую
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
            "id":     getattr(u, "id", None),
            "brand":  getattr(u, "brand", "") or "",
            "model":  getattr(u, "model", "") or "",
            "serial": getattr(u, "serial", "") or "",
            "rows":   rows,
        })

    # если у заказа ещё нет юнитов — создаём один из шапки/пустой
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

    # Важно: всегда рендерим multi-форму (есть кнопка "+ Add appliance")
    return render_template(
        "wo_form_units.html",
        wo=wo,
        units=units,
        recent_suppliers=recent_suppliers,
        readonly=readonly,
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
            current_app.logger.warning("Model delete failed, fallback to SQL: %s", model_err)

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


@inventory_bp.post("/work_orders/savex")
@login_required
def wo_savex():
    if getattr(current_user, "role", "") not in ("admin", "superadmin"):
        flash("Access denied", "danger")
        return redirect(url_for("inventory.wo_list"))

    from models import WorkOrder, WorkUnit, WorkOrderPart, Part
    from extensions import db
    from sqlalchemy import func

    f = request.form

    # helpers
    def _tech_norm(x: str) -> str:
        return (x or "").strip().upper()

    def _f(x, default=0.0):
        try:
            return float(x)
        except Exception:
            return float(default)

    def _i(x, default=0):
        try:
            return int(x)
        except Exception:
            return int(default)

    def _clip(s: str, n: int) -> str:
        return (s or "").strip()[:n]

    # header
    wo_id        = (f.get("wo_id") or "").strip()
    tech         = _tech_norm(f.get("technician_name"))
    job_numbers  = (f.get("job_numbers") or "").strip()
    brand_hdr    = (f.get("brand") or "").strip()
    model_hdr    = _clip(f.get("model"), 25)
    serial_hdr   = _clip(f.get("serial"), 25)
    job_type     = (f.get("job_type") or "BASE").strip().upper()
    status_hdr   = (f.get("status") or "search_ordered").strip()
    delivery_fee   = _f(f.get("delivery_fee"), 0)
    markup_percent = _f(f.get("markup_percent"), 0)

    # ---------- 1) ПАРСИМ PAYLOAD (без изменений БД) ----------
    units_payload = []
    try:
        # если есть ваш универсальный парсер — используем его
        from inventory.utils import _parse_units_form
        units_payload = _parse_units_form(f) or []
    except Exception:
        units_payload = []

    # Fallback — однотабличная форма
    if not units_payload:
        rows = []
        idx = 0
        while True:
            key = f"rows[{idx}][part_number]"
            if key not in f:
                break

            pn = (f.get(key) or "").strip().upper()
            qty = _i(f.get(f"rows[{idx}][quantity]") or 0, 0)

            alt_raw = (
                (f.get(f"rows[{idx}][alt_pn]") or "") or
                (f.get(f"rows[{idx}][alt_numbers]") or "") or
                (f.get(f"rows[{idx}][alt_part_numbers]") or "")
            ).strip()

            unit_cost = f.get(f"rows[{idx}][unit_cost]")
            try:
                unit_cost = float(unit_cost) if (unit_cost not in (None, "")) else None
            except Exception:
                unit_cost = None

            total_hint = f.get(f"rows[{idx}][line_total]")
            try:
                total_hint = float(total_hint) if (total_hint not in (None, "")) else None
            except Exception:
                total_hint = None

            # ВАЖНО: читаем ordered_flag из скрытого поля (0/1)
            ordered_flag_raw = f.get(f"rows[{idx}][ordered_flag]")
            try:
                ordered_flag = bool(int(ordered_flag_raw))
            except Exception:
                ordered_flag = False

            row = {
                "part_number": pn,
                "part_name": (f.get(f"rows[{idx}][part_name]") or "").strip(),
                "quantity": qty,
                "alt_numbers": alt_raw,
                "warehouse": (f.get(f"rows[{idx}][warehouse]") or "").strip(),
                "supplier":  (f.get(f"rows[{idx}][supplier]")  or "").strip(),
                "unit_cost": unit_cost,
                "total_hint": total_hint,
                "backorder_flag": bool(f.get(f"rows[{idx}][backorder_flag]")),
                "stock_hint": (f.get(f"rows[{idx}][stock_hint]") or "").strip().upper(),
                "issue_flag": bool(f.get(f"rows[{idx}][issue_flag]")),
                "ordered_flag": ordered_flag,  # ← добавили
            }
            if pn and qty > 0:
                rows.append(row)
            idx += 1

        if rows or any([brand_hdr, model_hdr, serial_hdr]):
            units_payload = [{
                "brand": brand_hdr,
                "model": model_hdr,
                "serial": serial_hdr,
                "rows": rows
            }]

    # Если в шапке указаны brand/model/serial — дублируем в первый юнит
    if any([brand_hdr, model_hdr, serial_hdr]) and units_payload:
        first = units_payload[0]
        if not ((first.get("brand") or "").strip() == brand_hdr and
                (first.get("model") or "").strip() == model_hdr and
                (first.get("serial") or "").strip() == serial_hdr):
            units_payload = [{"brand": brand_hdr, "model": model_hdr, "serial": serial_hdr, "rows": []}] + units_payload

    def _first_nonempty(lst, key):
        for it in lst or []:
            v = (it.get(key) or "").strip()
            if v:
                return v
        return ""

    # ---------- 2) СЕРВЕРНАЯ ВАЛИДАЦИЯ SUP (обязательное) ----------
    def _collect_missing_sup(payload):
        missing = []
        for u in payload or []:
            for r in (u.get("rows") or []):
                pn  = (r.get("part_number") or "").strip().upper()
                qty = _i(r.get("quantity") or 0, 0)
                sup = (r.get("supplier") or "").strip()
                if pn and qty > 0 and not sup:
                    missing.append(pn)
        return sorted(set(missing))

    missing_pns = _collect_missing_sup(units_payload)
    if missing_pns:
        flash(f"Supplier (Sup) is required for: {', '.join(missing_pns)}", "danger")
        if wo_id:
            return redirect(url_for("inventory.wo_edit", wo_id=int(wo_id)))
        else:
            return redirect(url_for("inventory.wo_new"))

    # ---------- 3) СОХРАНЕНИЕ В БД ----------
    if wo_id:
        wo = WorkOrder.query.get_or_404(int(wo_id))
    else:
        wo = WorkOrder()
        db.session.add(wo)

    wo.technician_name = tech
    wo.job_numbers     = job_numbers
    wo.job_type        = job_type
    wo.status          = status_hdr if status_hdr in ("search_ordered", "ordered", "done") else "search_ordered"
    wo.delivery_fee    = delivery_fee
    wo.markup_percent  = markup_percent

    wo.brand  = _clip(_first_nonempty(units_payload, "brand"), 40)
    wo.model  = _clip(_first_nonempty(units_payload, "model"), 25)
    wo.serial = _clip(_first_nonempty(units_payload, "serial"), 25)

    # чистая пересборка юнитов
    if wo_id:
        for u in list(getattr(wo, "units", []) or []):
            db.session.delete(u)
        db.session.flush()

    # сохраняем
    for u in units_payload or []:
        has_meta = any([(u.get("brand") or "").strip(),
                        (u.get("model") or "").strip(),
                        (u.get("serial") or "").strip()])
        rows = u.get("rows") or []
        if not has_meta and not rows:
            continue

        unit = WorkUnit(
            work_order=wo,
            brand=(u.get("brand") or "").strip(),
            model=_clip(u.get("model"), 25),
            serial=_clip(u.get("serial"), 25),
        )
        db.session.add(unit)
        db.session.flush()

        for r in rows:
            pn = _clip((r.get("part_number") or "").upper(), 80)
            if not pn:
                continue
            qty = _i(r.get("quantity") or 0, 0)

            uc = r.get("unit_cost")
            try:
                unit_cost = float(uc) if (uc not in (None, "")) else None
            except Exception:
                unit_cost = None

            alt_raw = (r.get("alt_numbers") or r.get("alt_part_numbers") or "")
            tokens = [_clip(t, 20) for t in (alt_raw.split(",") if isinstance(alt_raw, str) else [])]
            alt_joined = ",".join([t for t in tokens if t])

            part_rec   = Part.query.filter(func.upper(Part.part_number) == pn).first()
            part_name  = _clip(r.get("part_name") or (getattr(part_rec, "name", None) or ""), 120)
            wh_from_db = getattr(part_rec, "location", None) or getattr(part_rec, "wh", None) or ""
            warehouse  = _clip(r.get("warehouse") or wh_from_db, 120)

            supplier_val = _clip((r.get("supplier") or "").strip(), 80)

            # ВАЖНО: ordered_flag. Сохраняем только когда общий статус 'ordered'
            ordered_flag_in = bool(int(r.get("ordered_flag") or 0))
            ordered_flag_final = ordered_flag_in if (wo.status == "ordered") else False

            wop = WorkOrderPart(
                work_order=wo,
                unit=unit,
                part_number=pn,
                part_name=part_name,
                quantity=qty,
                alt_numbers=alt_joined,
                warehouse=warehouse,
                supplier=supplier_val,
                backorder_flag=bool(r.get("backorder_flag")),
            )

            if hasattr(wop, "unit_cost"):
                wop.unit_cost = unit_cost

            # Сохраняем ordered флаг/статус
            if hasattr(wop, "ordered_flag"):
                wop.ordered_flag = ordered_flag_final
            # если отдельного поля нет — маппим в line_status
            if hasattr(wop, "line_status"):
                wop.line_status = "ordered" if ordered_flag_final else "search_ordered"

            db.session.add(wop)

    db.session.commit()
    flash("Work Order saved.", "success")
    return redirect(url_for("inventory.wo_detail", wo_id=wo.id))

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
            items.append({"part_id": part.id, "qty": int(r["issue_now"]), "unit_price": None})

    if not items:
        flash("Nothing available to issue for this unit.", "warning")
        return redirect(url_for("inventory.wo_detail", wo_id=wo_id))

    # --- создаём строки выдачи ---
    issue_date, maybe_records = _issue_records_bulk(
        issued_to=wo.technician_name,
        reference_job=wo.canonical_job,
        items=items
    )

    # поддержка обоих вариантов возврата
    if isinstance(maybe_records, (list, tuple)) and maybe_records and hasattr(maybe_records[0], "id"):
        new_records = list(maybe_records)
    else:
        # подбираем только что созданные строки окном по времени
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
        # даже если не собрали записи — статус всё равно поменяем, если просили
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

    # --- обновим статус WO ---
    if set_status == "done":
        try:
            wo.status = "done"
            db.session.commit()
        except Exception:
            db.session.rollback()
    else:
        try:
            avail_all = compute_availability_multi(wo)
            still_wait = any(int(x.get("on_hand", 0)) < int(x.get("requested", 0)) for x in avail_all)
            if not still_wait:
                wo.status = "done"
                db.session.commit()
        except Exception:
            db.session.rollback()

    # --- редирект прямо на отчёт ---
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
    from extensions import db
    from models import IssuedPartRecord, IssuedBatch, Part
    from flask import render_template, request

    def _parse_date_ymd(s: str):
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
    recipient      = (params.get('recipient') or '').strip() or None
    reference_job  = (params.get('reference_job') or '').strip() or None
    # принимаем invoice_number, а также fallback: invoice / invoice_no
    invoice_s      = (params.get('invoice_number') or params.get('invoice') or params.get('invoice_no') or '').strip()
    location       = (params.get('location') or '').strip() or None

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
        IssuedPartRecord.query
        .join(Part, IssuedPartRecord.part_id == Part.id)
        .options(
            selectinload(IssuedPartRecord.part),
            selectinload(IssuedPartRecord.batch),
        )
    )

    if recipient:
        q = q.filter(IssuedPartRecord.issued_to.ilike(f'%{recipient}%'))
    if reference_job:
        q = q.filter(IssuedPartRecord.reference_job.ilike(f'%{reference_job}%'))
    if start_dt:
        q = q.filter(IssuedPartRecord.issue_date >= start_dt)
    if end_dt:
        q = q.filter(IssuedPartRecord.issue_date <= end_dt)
    if invoice_no is not None:
        # у legacy строк invoice_number может быть NULL — показываем только совпавшие
        q = q.filter(IssuedPartRecord.invoice_number == invoice_no)
    if location:
        q = q.filter(IssuedPartRecord.location == location)

    rows = q.order_by(
        IssuedPartRecord.issue_date.desc(),
        IssuedPartRecord.id.desc()
    ).all()

    # ---------- Группировка: новые по batch_id, legacy по старому ключу ----------
    grouped = defaultdict(list)
    for r in rows:
        if getattr(r, 'batch_id', None):
            key = ('BATCH', r.batch_id)
        else:
            key = ('LEGACY', r.issued_to, r.reference_job, r.issued_by, r.issue_date.date())
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
        items_sorted = sorted(items, key=lambda it: it.id)  # стабильный порядок строк
        total_value = sum((it.quantity or 0) * (it.unit_cost_at_issue or 0.0) for it in items_sorted)
        grand_total += total_value

        if gkey[0] == 'BATCH':
            batch = items_sorted[0].batch  # selectinload уже сделал
            inv = {
                'id': f'B{batch.id}',
                'issued_to': batch.issued_to,
                'reference_job': batch.reference_job,
                'issued_by': batch.issued_by,
                'issue_date': batch.issue_date,          # datetime
                'invoice_number': batch.invoice_number,
                'location': batch.location,
                'items': items_sorted,
                'total_value': total_value,
                'is_return': _is_return_records(items_sorted),
                '_sort_dt': max((it.issue_date for it in items_sorted if it.issue_date), default=batch.issue_date),
                '_sort_id': max((it.id for it in items_sorted), default=0),
            }
        else:
            _, issued_to, ref_job, issued_by, day = gkey
            issue_dt = datetime.combine(day, time.min)
            first = items_sorted[0]
            inv = {
                'id': f'K{issued_to}|{issued_by}|{ref_job or ""}|{day.isoformat()}',
                'issued_to': issued_to,
                'reference_job': ref_job,
                'issued_by': issued_by,
                'issue_date': issue_dt,                  # datetime
                'invoice_number': first.invoice_number,  # может быть None (legacy)
                'location': first.location,
                'items': items_sorted,
                'total_value': total_value,
                'is_return': _is_return_records(items_sorted),
                '_sort_dt': max((it.issue_date for it in items_sorted if it.issue_date), default=issue_dt),
                '_sort_id': max((it.id for it in items_sorted), default=0),
            }

        invoices.append(inv)

    # ---------- Сортировка карточек (НОВАЯ) ----------
    # Только по дате (последние сверху), затем по номеру инвойса, затем по макс. id строки.
    invoices.sort(
        key=lambda g: (
            g.get('_sort_dt') or datetime.min,
            g.get('invoice_number') or 0,
            g.get('_sort_id') or 0,
        ),
        reverse=True
    )

    return render_template(
        'reports_grouped.html',
        invoices=invoices,
        total=grand_total,
        # возвращаем исходные значения в форму (не очищаем)
        start_date=start_date_s,
        end_date=end_date_s,
        recipient=recipient or '',
        reference_job=reference_job or '',
        invoice=invoice_s or '',  # важно: шаблон ждёт 'invoice', не 'invoice_number'
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
    from sqlalchemy import func
    from flask import request, make_response

    # ---- вспомогалки (локальные, чтобы ничего не поломать) ----
    def _next_invoice_number():
        mb = db.session.query(func.coalesce(func.max(IssuedBatch.invoice_number), 0)).scalar() or 0
        ml = db.session.query(func.coalesce(func.max(IssuedPartRecord.invoice_number), 0)).scalar() or 0
        return max(int(mb), int(ml)) + 1

    def _ensure_invoice_number_for_records(records, issued_to, issued_by, reference_job, issue_date, location):
        # уже есть — выходим
        if any(getattr(r, "invoice_number", None) for r in records):
            return getattr(records[0], "invoice_number", None)

        # пробуем штатный хелпер, если он у тебя есть
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
                    db.session.flush()  # держим уникальность

                    for r in records:
                        r.batch_id = batch.id
                        r.invoice_number = inv_no

                    db.session.flush()
                return inv_no
            except Exception:
                db.session.rollback()
                continue
        raise RuntimeError("Failed to reserve invoice number")

    # ---- входные параметры ----
    inv_s  = (request.args.get("invoice_number") or "").strip()
    issued_to     = (request.args.get("issued_to") or "").strip()
    reference_job = (request.args.get("reference_job") or "").strip() or None
    issued_by     = (request.args.get("issued_by") or "").strip()
    issue_date_s  = (request.args.get("issue_date") or "").strip()

    inv_no = int(inv_s) if inv_s.isdigit() else None

    # ---- загрузка строк группы ----
    recs = []
    if inv_no is not None:
        recs = IssuedPartRecord.query.filter_by(invoice_number=inv_no).order_by(IssuedPartRecord.id.asc()).all()
    else:
        # legacy-поиск по ключам за сутки (как в update_invoice)
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

    # ---- если у группы ещё нет номера — присваиваем его перед печатью ----
    if all(getattr(r, "invoice_number", None) is None for r in recs):
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
        except Exception as e:
            db.session.rollback()
            # если не получилось — печатаем read-only как есть (legacy), но пользователь хотя бы получит PDF
            inv_no = None

    # ---- генерим PDF (используем твой генератор) ----
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

    def _ensure_invoice_number_for_records(records, issued_to, issued_by, reference_job, issue_date, location):
        """
        Ensure a single invoice_number is assigned to all given records.
        Try a project helper `_create_batch_for_records` if it exists; otherwise use a safe fallback.
        """
        if any(getattr(r, "invoice_number", None) for r in records):
            return

        # Try your project helper if present
        try:
            batch = _create_batch_for_records(
                records=records,
                issued_to=issued_to,
                issued_by=issued_by,
                reference_job=reference_job,
                issue_date=issue_date,
                location=location
            )
            return batch
        except Exception:
            db.session.rollback()

        # Fallback: reserve a unique number inside a nested transaction
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
                    db.session.flush()  # reserve unique number

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

        Rule (robust):
          - match by the same part (part_id),
          - same receiver (issued_to),
          - reference_job is exactly 'RETURN <r.reference_job>' when r.reference_job is set,
            otherwise ANY 'RETURN%' (legacy empty ref),
          - quantity < 0 (returns).
        We DO NOT depend on invoice_number or issued_by — returns may later receive
        their own number / be performed by another user.
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

        # reference match
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

                # How many units are still eligible to be returned for this issued row
                available = _available_to_return_for(r)
                if available <= 0:
                    # Nothing left to return for this line
                    continue

                # Requested quantity from the form (defensive parsing)
                raw = (request.form.get(f"qty_{r.id}") or "1").strip()
                try:
                    qty_req = int(raw)
                except Exception:
                    qty_req = 1

                if qty_req < 0:
                    qty_req = 0

                # Hard cap: do not allow exceeding available remaining quantity
                if qty_req > available:
                    qty_req = available
                    trimmed_any = True

                # If after trimming there's nothing to return — skip
                if qty_req <= 0:
                    continue

                # Create a negative "return" row; reference_job mirrors how we generate returns elsewhere
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
                    invoice_number=None,  # returns don't receive a number here
                    batch_id=None  # and are not attached to a batch
                )
                db.session.add(ret)

                # Stock: returns increase stock by the returned quantity
                if r.part:
                    r.part.quantity = int(r.part.quantity or 0) + qty_req

                created += 1

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

@inventory_bp.route('/import', methods=['GET', 'POST'])
@login_required
def import_parts():
    current_app.logger.warning(">>> NEW /import ROUTE HIT (PDF enabled)")
    enabled = current_app.config.get("WCCR_IMPORT_ENABLED", 0)
    dry     = current_app.config.get("WCCR_IMPORT_DRY_RUN", 1)

    # --- GET: show upload form ---
    if request.method == "GET":
        return render_template("import_parts.html")

    # --- POST: two flows -> save_preview / apply OR first upload ---
    # 1) User pressed "Save edits" in preview
    if "save_preview" in request.form:
        saved_path = (request.form.get("saved_path") or "").strip()
        rows_edited = parse_preview_rows(request.form)
        df_user = rows_to_dataframe(rows_edited)

        # Normalize again to keep schema + row_key consistent
        fname = os.path.basename(saved_path).lower()
        hint = None
        if re.search(r"\breliable\b", fname):
            hint = "ReliableParts"
        elif re.search(r"\bmarcone\b", fname):
            hint = "Marcone"

        norm, issues = normalize_table(
            df_user,  # см. пункт 2 ниже
            supplier_hint=hint,
            source_file=saved_path,
            default_location="MAIN"
        )

        # <<< LOG: что именно распознали

        current_app.logger.warning(
            "IMPORT/PREVIEW: supplier_hint=%s file=%s rows_out=%d | in_cols=%s",
            hint, os.path.basename(saved_path), len(norm), list(df_user.columns)
        )
        if len(norm) == 0:
            # полезно увидеть первые строки сырого df, если ничего не распозналось
            current_app.logger.warning("IMPORT/PREVIEW: df_user head:\n%s", df_user.head(10).to_string())

        for msg in issues:
            flash(msg, "warning")

        flash("Edits saved in preview (not applied yet).", "info")
        return render_template("import_preview.html", rows=norm.to_dict(orient="records"), saved_path=saved_path)

    # 2) User pressed "Apply import" in preview
    if "apply" in request.form:
        saved_path = (request.form.get("saved_path") or "").strip()
        if not saved_path or not os.path.exists(saved_path):
            flash("Saved file not found. Upload it again.", "danger")
            return redirect(url_for("inventory.import_parts"))

        # Prefer edited rows from the form if present
        rows_edited = parse_preview_rows(request.form)
        if rows_edited:
            df = rows_to_dataframe(rows_edited)
        else:
            # fallback: re-read file
            ext = os.path.splitext(saved_path)[1].lower()
            if ext == ".pdf":
                df = dataframe_from_pdf(saved_path, try_ocr=True)
                df = coerce_invoice_items(df)
            elif ext in {".xlsx", ".xls", ".csv"}:
                df = load_table(saved_path)
            else:
                flash("Unsupported file type. Use .pdf, .xlsx, .xls or .csv", "warning")
                return redirect(url_for("inventory.import_parts"))

        fname = os.path.basename(path).lower()
        supplier_hint = "ReliableParts" if "reliable" in fname else ("Marcone" if "marcone" in fname else None)

        norm, issues = normalize_table(
            df,
            supplier_hint=supplier_hint,
            source_file=path,
            default_location="MAIN",
        )
        for msg in issues:
            flash(msg, "warning")

        rows = norm.to_dict(orient="records")
        return render_template("import_preview.html", rows=rows, saved_path=path)

        # --- Apply: build and execute RECEIVE movements ---
        def duplicate_exists(rk: str) -> bool:
            return has_key(rk)

        def make_movement(m: dict) -> None:
            PartModel = Part
            session = db.session
            PN_FIELDS   = ["part_number","number","sku","code","partnum","pn"]
            NAME_FIELDS = ["name","part_name","descr","description","title"]
            QTY_FIELDS  = ["quantity","qty","on_hand","stock","count"]
            LOC_FIELDS  = ["location","bin","shelf","place","loc"]
            COST_FIELDS = ["unit_cost","cost","price","unitprice","last_cost"]
            SUP_FIELDS  = ["supplier","vendor","provider"]

            def pick_field(model, candidates):
                for f in candidates:
                    if hasattr(model, f): return f
                return None

            pn_field   = pick_field(PartModel, PN_FIELDS)
            name_field = pick_field(PartModel, NAME_FIELDS)
            qty_field  = pick_field(PartModel, QTY_FIELDS)
            loc_field  = pick_field(PartModel, LOC_FIELDS)
            cost_field = pick_field(PartModel, COST_FIELDS)
            sup_field  = pick_field(PartModel, SUP_FIELDS)
            if pn_field is None or qty_field is None:
                raise RuntimeError("Part model is missing PART # or QTY field.")

            filters = {pn_field: m["part_number"]}
            if loc_field:
                filters[loc_field] = m["location"]
            part = PartModel.query.filter_by(**filters).first()

            if not part:
                kwargs = dict(filters)
                kwargs[qty_field] = 0
                if name_field and m["part_name"]: kwargs[name_field] = m["part_name"]
                if cost_field and (m["unit_cost"] is not None): kwargs[cost_field] = float(m["unit_cost"])
                if sup_field and m.get("supplier"): kwargs[sup_field] = m["supplier"]
                part = PartModel(**kwargs)
                session.add(part)
                session.flush()

            # update fields
            if name_field and not getattr(part, name_field) and m["part_name"]:
                setattr(part, name_field, m["part_name"])
            if cost_field and (m["unit_cost"] is not None):
                setattr(part, cost_field, float(m["unit_cost"]))
            setattr(part, qty_field, (getattr(part, qty_field) or 0) + int(m["qty"]))
            session.commit()
            add_key(m["row_key"], {"file": m["source_file"]})

        built, errors = build_receive_movements(norm, duplicate_exists_func=duplicate_exists, make_movement_func=make_movement)
        for e in errors:
            flash(e, "danger")
        flash(f"Created receipts: {len(built)}", "success")
        try:
            os.remove(saved_path)
        except Exception:
            pass
        return redirect(url_for("inventory.import_parts"))

    # 3) First upload → build preview (no edits yet)
    f = request.files.get("file")
    if not f or f.filename == "":
        flash("Choose a .pdf, .xlsx, .xls, or .csv file.", "warning")
        return redirect(request.url)

    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in {".pdf", ".xlsx", ".xls", ".csv"}:
        flash("Choose a .pdf, .xlsx, .xls, or .csv file.", "warning")
        return redirect(request.url)

    filename = secure_filename(f.filename)
    upload_dir = current_app.config.get("UPLOAD_FOLDER", os.path.join(current_app.instance_path, "uploads"))
    os.makedirs(upload_dir, exist_ok=True)
    path = os.path.join(upload_dir, filename)
    f.save(path)

    if ext == ".pdf":
        df = dataframe_from_pdf(path, try_ocr=True)
        df = coerce_invoice_items(df)
    else:
        df = load_table(path)

    fname = os.path.basename(path).lower()
    supplier_hint = "ReliableParts" if "reliable" in fname else ("Marcone" if "marcone" in fname else None)
    norm, issues = normalize_table(df, supplier_hint=supplier_hint, source_file=path, default_location="MAIN")
    for msg in issues:
        flash(msg, "warning")

    return render_template("import_preview.html", rows=norm.to_dict(orient="records"), saved_path=path)

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


@inventory_bp.route('/users')
@login_required
def users():
    # superadmin видит всех, admin — только пользователей с ролью user и себя
    if current_user.role == 'superadmin':
        users_list = User.query.all()
    elif current_user.role == 'admin':
        users_list = User.query.filter(
            (User.role == 'user') | (User.id == current_user.id)
        ).all()
    else:
        flash("Access denied", "danger")
        return redirect(url_for('inventory.dashboard'))

    return render_template('users.html', users=users_list)


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
    user = User.query.get_or_404(user_id)

    # superadmin может редактировать всех
    # admin — только пользователей с ролью user и себя
    if current_user.role == 'admin':
        if user.role != 'user' and user.id != current_user.id:
            flash("Admins can only edit users with role 'user' or themselves.", "danger")
            return redirect(url_for('inventory.dashboard'))
    elif current_user.role != 'superadmin':
        flash("Access denied", "danger")
        return redirect(url_for('inventory.dashboard'))

    if request.method == 'POST':
        user.username = request.form['username'].strip()
        user.role = request.form['role']
        db.session.commit()
        flash("User updated successfully", "success")
        return redirect(url_for('inventory.users'))

    return render_template('edit_user.html', user=user)


@inventory_bp.route('/users/change_password/<int:user_id>', methods=['GET', 'POST'])
@login_required
def change_password(user_id):
    user = User.query.get_or_404(user_id)

    # superadmin меняет любой пароль
    # admin меняет только пароли пользователей с ролью user и свой (для своего - проверка текущего пароля)
    if current_user.role == 'admin':
        if user.role != 'user' and user.id != current_user.id:
            flash("Admins can only change passwords for users with role 'user' or themselves.", "danger")
            return redirect(url_for('inventory.dashboard'))
    elif current_user.role != 'superadmin' and current_user.id != user_id:
        flash("Access denied", "danger")
        return redirect(url_for('inventory.dashboard'))

    if request.method == 'POST':
        # Если admin меняет свой пароль - проверяем текущий пароль
        if current_user.role == 'admin' and current_user.id == user_id:
            current_password = request.form.get('current_password')
            if not user.check_password(current_password):
                flash("Current password is incorrect.", "danger")
                return redirect(url_for('inventory.change_password', user_id=user_id))

        new_password = request.form['password']
        confirm_password = request.form.get('confirm_password')

        if new_password != confirm_password:
            flash("Passwords do not match.", "danger")
            return redirect(url_for('inventory.change_password', user_id=user_id))

        user.password_hash = generate_password_hash(new_password)

        db.session.commit()
        flash("Password changed successfully", "success")

        if current_user.role == 'superadmin':
            return redirect(url_for('inventory.users'))
        else:
            return redirect(url_for('inventory.dashboard'))

    return render_template('change_password.html', user=user)


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

@inventory_bp.route("/import-parts", methods=["GET", "POST"], endpoint="import_parts_upload")
def import_parts_upload():
    enabled = current_app.config.get("WCCR_IMPORT_ENABLED", 0)
    dry     = current_app.config.get("WCCR_IMPORT_DRY_RUN", 1)

    # -------- 1) Нажали "Применить импорт" с предпросмотра --------
    if request.method == "POST" and "apply" in request.form:
        saved_path = request.form.get("saved_path", "")
        if not saved_path or not os.path.exists(saved_path):
            flash("Не найден сохранённый файл. Загрузите его заново.", "danger")
            return redirect(url_for("inventory.import_parts_upload"))

        ext = os.path.splitext(saved_path)[1].lower()

        # Читаем и нормализуем тот же файл (PDF/Excel/CSV)
        if ext == ".pdf":
            # ⬇️ единственное изменение — явно отключаем OCR
            df = dataframe_from_pdf(saved_path, try_ocr=False)
            if df.empty:
                flash("В этом PDF не удалось распознать таблицы (скорее всего скан).", "danger")
                rows = []
                return render_template("import_preview.html", rows=rows, saved_path=saved_path)
        else:
            df = load_table(saved_path)

        norm, issues = normalize_table(df, supplier_hint=None, source_file=saved_path, default_location="MAIN")
        for msg in issues:
            flash(msg, "warning")

        # Если DRY или выключено — просто показать предпросмотр снова
        if dry or not enabled:
            rows = norm.to_dict(orient="records")
            return render_template("import_preview.html", rows=rows, saved_path=saved_path)

        # Применяем: создаём приходы, подавляя дубли по row_key
        def duplicate_exists(rk: str) -> bool:
            return has_key(rk)

        def make_movement(m: dict) -> None:
            PartModel = Part
            session = db.session

            PN_FIELDS   = ["part_number", "number", "sku", "code", "partnum", "pn"]
            NAME_FIELDS = ["name", "part_name", "descr", "description", "title"]
            QTY_FIELDS  = ["quantity", "qty", "on_hand", "stock", "count"]
            LOC_FIELDS  = ["location", "bin", "shelf", "place", "loc"]
            COST_FIELDS = ["unit_cost", "cost", "price", "unitprice", "last_cost"]
            SUP_FIELDS  = ["supplier", "vendor", "provider"]

            def pick_field(model, candidates):
                for f in candidates:
                    if hasattr(model, f):
                        return f
                return None

            pn_field   = pick_field(PartModel, PN_FIELDS)
            name_field = pick_field(PartModel, NAME_FIELDS)
            qty_field  = pick_field(PartModel, QTY_FIELDS)
            loc_field  = pick_field(PartModel, LOC_FIELDS)
            cost_field = pick_field(PartModel, COST_FIELDS)
            sup_field  = pick_field(PartModel, SUP_FIELDS)

            if pn_field is None or qty_field is None:
                raise RuntimeError("Не найдено поле PART # или QTY в модели Part — уточни имена полей.")

            filters = {pn_field: m["part_number"]}
            if loc_field:
                filters[loc_field] = m["location"]

            part = PartModel.query.filter_by(**filters).first()

            if not part:
                kwargs = dict(filters)
                kwargs[qty_field] = 0
                if name_field and m["part_name"]:
                    kwargs[name_field] = m["part_name"]
                if cost_field and (m["unit_cost"] is not None):
                    kwargs[cost_field] = float(m["unit_cost"])
                if sup_field and m.get("supplier"):
                    kwargs[sup_field] = m["supplier"]
                part = PartModel(**kwargs)
                session.add(part)
                session.flush()

            if name_field and not getattr(part, name_field) and m["part_name"]:
                setattr(part, name_field, m["part_name"])
            if cost_field and (m["unit_cost"] is not None):
                setattr(part, cost_field, float(m["unit_cost"]))

            current_qty = getattr(part, qty_field) or 0
            setattr(part, qty_field, current_qty + int(m["qty"]))

            session.commit()
            add_key(m["row_key"], {"file": m["source_file"]})

        built, errors = build_receive_movements(
            norm,
            duplicate_exists_func=duplicate_exists,
            make_movement_func=make_movement
        )
        for e in errors:
            flash(e, "danger")
        flash(f"Создано приходов: {len(built)}", "success")
        return redirect(url_for("inventory.import_parts_upload"))

    # -------- 2) Первая загрузка файла → показать предпросмотр --------
    if request.method == "POST":
        f = request.files.get("file")
        if not f or f.filename == "":
            flash("Выберите файл (.pdf, .xlsx, .xls или .csv)", "warning")
            return redirect(request.url)

        filename = secure_filename(f.filename)
        upload_dir = current_app.config.get("UPLOAD_FOLDER", os.path.join(current_app.instance_path, "uploads"))
        os.makedirs(upload_dir, exist_ok=True)
        path = os.path.join(upload_dir, filename)
        f.save(path)

        ext = os.path.splitext(path)[1].lower()

        if ext == ".pdf":
            # ⬇️ и здесь — выключаем OCR
            df = dataframe_from_pdf(path, try_ocr=False)
            if df.empty:
                flash("В этом PDF не удалось распознать таблицы (скорее всего скан).", "danger")
                rows = []
                return render_template("import_preview.html", rows=rows, saved_path=path)
        else:
            df = load_table(path)

        norm, issues = normalize_table(df, supplier_hint=None, source_file=path, default_location="MAIN")
        for msg in issues:
            flash(msg, "warning")

        rows = norm.to_dict(orient="records")
        return render_template("import_preview.html", rows=rows, saved_path=path)

    # -------- 3) GET → показать твою форму загрузки --------
    return render_template("import_parts.html")


# ========= ORDERS: список/поиск =========
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
















