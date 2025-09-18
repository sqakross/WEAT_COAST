from sqlalchemy.exc import SQLAlchemyError
import traceback

from sqlalchemy import func
from models import IssuedPartRecord, WorkOrder, WorkOrderPart, TechReceiveLog
import json
# from extensions import db
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
from flask import (
    Blueprint, render_template, request, redirect, url_for,
    flash, send_file, jsonify, after_this_request,
    current_app,abort,                    # NEW
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

@inventory_bp.post("/work_orders/new", endpoint="wo_create")
@login_required
def wo_create():
    if getattr(current_user, "role", "") not in ("admin", "superadmin"):
        flash("Access denied", "danger")
        return redirect(url_for("inventory.wo_list"))

    f = request.form
    technician_name = (f.get("technician_name") or "").strip()
    job_numbers = (f.get("job_numbers") or "").strip()
    brand = (f.get("brand") or "").strip()
    model = (f.get("model") or "").strip()
    serial = (f.get("serial") or "").strip()
    job_type = (f.get("job_type") or "BASE").strip().upper()
    delivery_fee = float(f.get("delivery_fee") or 0)
    markup_percent = float(f.get("markup_percent") or 0)

    if not technician_name or not job_numbers:
        flash("Technician and Job(s) are required.", "danger")
        return redirect(url_for("inventory.wo_new"))

    # канонический job = максимальный из перечисленных чисел
    jobs = [j.strip() for j in job_numbers.replace(",", " ").split() if j.strip()]
    nums = []
    for j in jobs:
        try:
            nums.append(int(j))
        except:
            pass
    canonical_job = str(max(nums)) if nums else (jobs[0] if jobs else "")

    wo = WorkOrder(
        technician_name=technician_name,
        job_numbers=", ".join(jobs),
        canonical_job=canonical_job,
        brand=brand, model=model, serial=serial,
        job_type=job_type if job_type in ("BASE","INSURANCE") else "BASE",
        delivery_fee=delivery_fee,
        markup_percent=markup_percent,
        status="search_ordered",
    )
    db.session.add(wo)
    db.session.flush()

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
        pn = (row.get("part_number") or "").strip()
        if not pn:
            continue
        part = WorkOrderPart(
            work_order_id=wo.id,
            part_number=pn,
            alt_numbers=(row.get("alt_numbers") or "").strip(),
            part_name=(row.get("part_name") or "").strip(),
            quantity=int(row.get("quantity") or 0),
            supplier=(row.get("supplier") or "").strip(),
            backorder=(row.get("backorder") == "on"),
        )
        db.session.add(part)

    db.session.commit()
    flash("Work order created.", "success")
    return redirect(url_for("inventory.wo_detail", wo_id=wo.id))

@inventory_bp.get("/work_orders/new")
@login_required
def wo_new():
    if getattr(current_user, "role", "") not in ("admin", "superadmin"):
        flash("Access denied", "danger")
        return redirect(url_for("inventory.wo_list"))
    # форма с пустыми значениями и одной строкой parts
    return render_template("wo_form.html", wo=None, parts=[{}])

@inventory_bp.post("/work_orders/save")
@login_required
def wo_save():
    if getattr(current_user, "role", "") not in ("admin", "superadmin"):
        flash("Access denied", "danger")
        return redirect(url_for("inventory.wo_list"))

    form = request.form
    wo_id = form.get("wo_id")
    is_new = not wo_id

    # Создаём/берём заказ
    if is_new:
        wo = WorkOrder()
        db.session.add(wo)
    else:
        wo = WorkOrder.query.get_or_404(int(wo_id))

    # --- Основные поля заказа ---
    wo.technician_name = (form.get("technician_name") or "").strip()
    wo.job_numbers     = (form.get("job_numbers") or "").strip()
    wo.brand           = (form.get("brand") or "").strip()
    wo.model           = (form.get("model") or "").strip()[:25]
    wo.serial          = (form.get("serial") or "").strip()[:25]
    wo.job_type        = (form.get("job_type") or "BASE").strip().upper()

    try:
        wo.delivery_fee = float(form.get("delivery_fee") or 0)
    except ValueError:
        wo.delivery_fee = 0.0

    try:
        wo.markup_percent = float(form.get("markup_percent") or 0)
    except ValueError:
        wo.markup_percent = 0.0

    _status = (form.get("status") or "search_ordered").strip()
    wo.status = _status if _status in ("search_ordered", "ordered", "done") else "search_ordered"

    # --- Пересобираем позиции (проще и надёжнее) ---
    if not is_new:
        for p in list(wo.parts):
            db.session.delete(p)

    idx = 0
    while True:
        # Проверяем наличие ключа строки
        key = f"rows[{idx}][part_number]"
        if key not in form:
            break

        pn          = (form.get(f"rows[{idx}][part_number]") or "").strip().upper()
        part_name   = (form.get(f"rows[{idx}][part_name]") or "").strip()
        qty_str     = (form.get(f"rows[{idx}][quantity]") or "0").strip()
        alt_pns     = (form.get(f"rows[{idx}][alt_part_numbers]") or "").strip()
        unit_label  = (form.get(f"rows[{idx}][unit_label]") or "").strip()         # NEW
        supplier    = (form.get(f"rows[{idx}][supplier]") or "").strip()
        backorder   = bool(form.get(f"rows[{idx}][backorder_flag]"))
        line_status = (form.get(f"rows[{idx}][status]") or "search_ordered").strip()
        if line_status not in ("search_ordered", "ordered", "done"):
            line_status = "search_ordered"

        try:
            qty = int(qty_str)
        except ValueError:
            qty = 0

        # Пропускаем «пустые» строки
        if pn and qty > 0:
            wop = WorkOrderPart(
                part_number=pn,
                part_name=(part_name[:80] or None),
                quantity=qty,
                alt_part_numbers=(alt_pns[:200] or None),
                unit_label=(unit_label[:120] or None),     # NEW
                supplier=(supplier[:80] or None),
                backorder_flag=backorder,
                status=line_status,
            )
            wo.parts.append(wop)

        idx += 1

    db.session.commit()
    flash("Work Order saved.", "success")
    return redirect(url_for("inventory.wo_detail", wo_id=wo.id))

@inventory_bp.post("/work_orders/<int:wo_id>/edit", endpoint="wo_update")
@login_required
def wo_update(wo_id):
    if getattr(current_user, "role", "") not in ("admin", "superadmin"):
        flash("Access denied", "danger")
        return redirect(url_for("inventory.wo_detail", wo_id=wo_id))

    wo = WorkOrder.query.get_or_404(wo_id)
    f = request.form

    wo.technician_name = (f.get("technician_name") or "").strip()
    raw_jobs = (f.get("job_numbers") or "").strip()
    jobs = [j.strip() for j in raw_jobs.replace(",", " ").split() if j.strip()]
    wo.job_numbers = ", ".join(jobs)
    # пересчёт canonical
    nums = []
    for j in jobs:
        try: nums.append(int(j))
        except: pass
    wo.canonical_job = str(max(nums)) if nums else (jobs[0] if jobs else "")

    wo.brand = (f.get("brand") or "").strip()
    wo.model = (f.get("model") or "").strip()
    wo.serial = (f.get("serial") or "").strip()
    job_type = (f.get("job_type") or "BASE").strip().upper()
    wo.job_type = job_type if job_type in ("BASE","INSURANCE") else "BASE"
    wo.delivery_fee = float(f.get("delivery_fee") or 0)
    wo.markup_percent = float(f.get("markup_percent") or 0)

    # пересобираем parts
    WorkOrderPart.query.filter_by(work_order_id=wo.id).delete()

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
        pn = (row.get("part_number") or "").strip()
        if not pn:
            continue
        part = WorkOrderPart(
            work_order_id=wo.id,
            part_number=pn,
            alt_numbers=(row.get("alt_numbers") or "").strip(),
            part_name=(row.get("part_name") or "").strip(),
            quantity=int(row.get("quantity") or 0),
            supplier=(row.get("supplier") or "").strip(),
            backorder=(row.get("backorder") == "on"),
        )
        db.session.add(part)

    db.session.commit()
    flash("Work order updated.", "success")
    return redirect(url_for("inventory.wo_detail", wo_id=wo.id))

@inventory_bp.get("/api/stock_hint")
@login_required
def api_stock_hint():
    pn  = (request.args.get("pn") or "").strip().upper()
    try:
        qty = int((request.args.get("qty") or "0").strip() or 0)
    except ValueError:
        qty = 0

    on = get_on_hand(pn) if pn else 0
    if qty > 0 and on >= qty:
        hint = "STOCK"
    elif qty > 0:
        hint = f"WAIT {qty} stock {on}"
    else:
        hint = "—"

    return jsonify({"on_hand": on, "hint": hint})

@inventory_bp.get("/work_orders")
@login_required
def wo_list():
    q = WorkOrder.query.order_by(WorkOrder.created_at.desc()).limit(200).all()
    return render_template("wo_list.html", items=q)

@inventory_bp.get("/work_orders/<int:wo_id>")
@login_required
def wo_detail(wo_id):
    # from flask import current_app, render_template, request
    # import os, sqlite3, traceback

    # ---- твои прежние загрузки (оставь как есть) ----
    # Пример:
    # wo = load_work_order(wo_id)              # ОБЯЗАТЕЛЕН
    # parts = load_work_order_parts(wo_id)     # опционально
    # avail = compute_availability(wo_id)      # если есть

    # Защита: если каких-то переменных нет, дадим дефолты,
    # чтобы шаблон не падал и мы увидели интерфейс.
    if "wo" not in locals() or wo is None:
        # минимальный суррогат, чтобы страница нарисовалась
        wo = type("WO", (), {})()
        wo.id = wo_id
        wo.technician_name = "-"
        wo.job_numbers = "-"
        wo.canonical_job = "-"
        wo.brand = "-"
        wo.model = "-"
        wo.serial = "-"
        wo.job_type = "BASE"
        wo.status = "search_ordered"
        wo.delivery_fee = 0.0
        wo.markup_percent = 0.0
        wo.created_at = None

    if "parts" not in locals() or parts is None:
        parts = []
    if "avail" not in locals() or avail is None:
        avail = []

    # ---- suppliers / batches с защитой ----
    suppliers = []
    batches = []
    try:
        dbp = os.path.normpath(os.path.join(current_app.instance_path, "inventory.db"))
        current_app.logger.info("wo_detail DB_PATH = %s", dbp)
        with sqlite3.connect(dbp) as con:
            con.row_factory = sqlite3.Row
            try:
                q = """SELECT DISTINCT supplier
                       FROM v_work_order_parts_dto
                       WHERE wo_id=? AND supplier<>'' 
                       ORDER BY supplier COLLATE NOCASE"""
                suppliers = [r[0] for r in con.execute(q, (wo_id,)).fetchall()]
            except Exception as e:
                current_app.logger.error("suppliers query failed: %s", e)
                current_app.logger.error(traceback.format_exc())
                suppliers = []

        # временный стаб для батчей (заменишь на реальные)
        batches = [{
            "issued_at": "2025-09-16 14:23",
            "technician": "Jane Doe",
            "canonical_ref": str(wo_id),
            "items": 3,
            "report_id": f"RPT-{wo_id}",
        }]
    except Exception as e:
        current_app.logger.error("wo_detail aux failed: %s", e)
        current_app.logger.error(traceback.format_exc())
        suppliers = suppliers or []
        batches = batches or []

    # ---- Рендер с безопасными значениями ----
    return render_template(
        "work_orders/detail.html",   # <- если у тебя другой путь, поменяй здесь
        wo=wo, parts=parts, avail=avail,
        suppliers=suppliers, batches=batches
    )

@inventory_bp.post("/work_orders/<int:wo_id>/refresh-stock")
@login_required
def wo_refresh_stock(wo_id):
    # TODO: тут обнови on_hand/ordered_qty/eta из склада/поставщика
    # временно заглушка:
    return jsonify({"ok": True, "updated": 0})
@inventory_bp.post("/work_orders/<int:wo_id>/issue_instock")
@login_required
def wo_issue_instock(wo_id):
    if getattr(current_user, "role", "") not in ("admin","superadmin"):
        flash("Access denied", "danger")
        return redirect(url_for("inventory.wo_detail", wo_id=wo_id))

    wo = WorkOrder.query.get_or_404(wo_id)
    avail = compute_availability(wo)

    items = []
    for row in avail:
        if row["issue_now"] > 0:
            part = Part.query.filter_by(part_number=row["part_number"].upper()).first()
            if not part:
                continue
            items.append({"part_id": part.id, "qty": row["issue_now"], "unit_price": None})

    if not items:
        flash("Nothing available to issue (all WAIT).", "warning")
        return redirect(url_for("inventory.wo_detail", wo_id=wo_id))

    issue_date, created = _issue_records_bulk(
        issued_to=wo.technician_name,
        reference_job=wo.canonical_job,
        items=items
    )

    still_wait = any(r["on_hand"] < r["requested"] for r in avail)
    if not still_wait:
        wo.status = "done"
        db.session.commit()

    today = issue_date.date().isoformat()
    params = urlencode({"start_date":today,"end_date":today,
                        "recipient":wo.technician_name,"reference_job":wo.canonical_job})
    link = f"/reports_grouped?{params}"
    flash(Markup(f'Issued in-stock items. <a href="{link}">Open invoice group</a> to print.'), "success")
    return redirect(link, code=303)

@inventory_bp.get("/work_orders/newx")
@login_required
def wo_newx():
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
    return render_template("wo_form_units.html", wo=None, units=units)


@inventory_bp.get("/work_orders/<int:wo_id>/editx")
@login_required
def wo_editx(wo_id):
    if getattr(current_user, "role", "") not in ("admin", "superadmin"):
        flash("Access denied", "danger")
        return redirect(url_for("inventory.wo_list"))
    wo = WorkOrder.query.get_or_404(wo_id)
    units = []
    for u in wo.units or []:
        rows = []
        for p in u.parts or []:
            rows.append({
                "part_number": p.part_number,
                "part_name": p.part_name or "",
                "quantity": p.quantity or 0,
                "alt_numbers": p.alt_numbers or "",
                "supplier": p.supplier or "",
                "backorder_flag": bool(p.backorder_flag),
                "line_status": p.line_status or "search_ordered",
            })
        if not rows:
            rows = [{"part_number": "", "part_name": "", "quantity": 1,
                     "alt_numbers": "", "supplier": "", "backorder_flag": False, "line_status": "search_ordered"}]
        units.append({
            "id": u.id,
            "brand": u.brand or "",
            "model": u.model or "",
            "serial": u.serial or "",
            "rows": rows
        })

    if not units:
        units = [{
            "brand": "", "model": "", "serial": "",
            "rows": [{"part_number": "", "part_name": "", "quantity": 1,
                      "alt_numbers": "", "supplier": "", "backorder_flag": False, "line_status": "search_ordered"}]
        }]

    return render_template("wo_form_units.html", wo=wo, units=units)


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


@inventory_bp.post("/work_orders/savex")
@login_required
def wo_savex():
    if getattr(current_user, "role", "") not in ("admin", "superadmin"):
        flash("Access denied", "danger")
        return redirect(url_for("inventory.wo_list"))

    from models import WorkUnit, WorkUnitPart  # импортируй из своего модуля моделей

    wo_id = (request.form.get("wo_id") or "").strip()
    # общие поля WO — возьмём как у твоей старой формы
    tech = (request.form.get("technician_name") or "").strip()
    job_numbers = (request.form.get("job_numbers") or "").strip()
    brand = (request.form.get("brand") or "").strip()
    model = (request.form.get("model") or "").strip()
    serial = (request.form.get("serial") or "").strip()
    job_type = (request.form.get("job_type") or "BASE").strip()
    status   = (request.form.get("status") or "search_ordered").strip()
    delivery_fee   = float(request.form.get("delivery_fee") or 0)
    markup_percent = float(request.form.get("markup_percent") or 0)

    # если поля brand/model/serial заполнены сверху — создадим для них отдельный unit №0
    has_header_unit = any([brand, model, serial])

    # units с вложенными rows
    units = _parse_units_form(request.form)
    if has_header_unit:
        units = [{"brand": brand, "model": model, "serial": serial, "rows": []}] + units

    # создать/обновить WO
    if wo_id:
        wo = WorkOrder.query.get_or_404(int(wo_id))
    else:
        wo = WorkOrder()
        db.session.add(wo)

    wo.technician_name = tech
    wo.job_numbers     = job_numbers
    wo.job_type        = job_type
    wo.status          = status
    wo.delivery_fee    = delivery_fee
    wo.markup_percent  = markup_percent

    # очистим старые units и создадим заново (проще и безопаснее)
    for u in (wo.units or []):
        db.session.delete(u)
    db.session.flush()

    # создаём units + parts
    for u in units:
        if not (u.get("brand") or u.get("model") or u.get("serial") or u.get("rows")):
            continue  # пустые секции пропускаем
        unit = WorkUnit(order=wo,
                        brand=u.get("brand") or "",
                        model=u.get("model") or "",
                        serial=u.get("serial") or "")
        db.session.add(unit)
        db.session.flush()

        for r in u.get("rows") or []:
            if not r.get("part_number"):
                continue
            part = WorkUnitPart(
                unit=unit,
                part_number=(r.get("part_number") or "").upper(),
                part_name=r.get("part_name") or "",
                quantity=int(r.get("quantity") or 0),
                alt_numbers=r.get("alt_numbers") or "",
                supplier=r.get("supplier") or "",
                backorder_flag=bool(r.get("backorder_flag")),
                line_status=(r.get("line_status") or "search_ordered"),
            )
            db.session.add(part)

    db.session.commit()
    flash("Work Order saved.", "success")
    return redirect(url_for("inventory.wo_detail", wo_id=wo.id))

@inventory_bp.post("/work_orders/<int:wo_id>/units/<int:unit_id>/issue_instock")
@login_required
def wo_issue_instock_unit(wo_id, unit_id):
    if getattr(current_user, "role", "") not in ("admin","superadmin"):
        flash("Access denied", "danger")
        return redirect(url_for("inventory.wo_detail", wo_id=wo_id))

    wo = WorkOrder.query.get_or_404(wo_id)
    unit = next((u for u in (wo.units or []) if u.id == unit_id), None)
    if not unit:
        flash("Unit not found", "danger")
        return redirect(url_for("inventory.wo_detail", wo_id=wo_id))

    rows = compute_availability_unit(unit, wo.status)

    items = []
    for r in rows:
        if r["issue_now"] > 0:
            part = Part.query.filter_by(part_number=r["part_number"].upper()).first()
            if not part:
                continue
            items.append({"part_id": part.id, "qty": r["issue_now"], "unit_price": None})

    if not items:
        flash("Nothing available to issue for this unit.", "warning")
        return redirect(url_for("inventory.wo_detail", wo_id=wo_id))

    issue_date, created = _issue_records_bulk(
        issued_to=wo.technician_name,
        reference_job=wo.canonical_job,   # твоя текущая логика primary job
        items=items
    )

    # Проверим, все ли закрыто по ВСЕМ units → done
    avail_all = compute_availability_multi(wo)
    still_wait = any(x["on_hand"] < x["requested"] for x in avail_all)
    if not still_wait:
        wo.status = "done"
        db.session.commit()

    today = issue_date.date().isoformat()
    params = urlencode({"start_date": today, "end_date": today,
                        "recipient": wo.technician_name, "reference_job": wo.canonical_job})
    return redirect(f"/reports_grouped?{params}", code=303)

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

@inventory_bp.route('/reports_grouped', methods=['GET', 'POST'])
@login_required
def reports_grouped():
    query = IssuedPartRecord.query.join(Part)

    # ← читаем и из GET (?start_date=...) и из POST-форм
    params = request.values
    start_date = params.get('start_date')
    end_date = params.get('end_date')
    recipient = params.get('recipient')
    reference_job = params.get('reference_job')

    # ← ВАЖНО: приводим строки дат к datetime, чтобы сравнивать корректно
    if start_date:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        query = query.filter(IssuedPartRecord.issue_date >= start_dt)

    if end_date:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(
            hour=23, minute=59, second=59, microsecond=999999
        )
        query = query.filter(IssuedPartRecord.issue_date <= end_dt)

    if recipient:
        query = query.filter(IssuedPartRecord.issued_to.ilike(f'%{recipient}%'))
    if reference_job:
        query = query.filter(IssuedPartRecord.reference_job.ilike(f'%{reference_job}%'))

    records = query.order_by(IssuedPartRecord.issue_date.desc()).all()

    # Группировка — как у тебя было
    grouped = defaultdict(list)
    for r in records:
        key = (r.issued_to, r.reference_job, r.issued_by, r.issue_date.date())
        grouped[key].append(r)

    invoices = []
    grand_total = 0
    for key, items in grouped.items():
        total_value = sum(item.quantity * item.unit_cost_at_issue for item in items)
        grand_total += total_value
        invoices.append({
            'issued_to': key[0],
            'reference_job': key[1],
            'issued_by': key[2],
            'issue_date': key[3],
            'items': items,
            'total_value': total_value
        })

    return render_template(
        'reports_grouped.html',
        invoices=invoices,
        total=grand_total,
        start_date=start_date,
        end_date=end_date,
        recipient=recipient,
        reference_job=reference_job
    )


@inventory_bp.route('/invoice/view')
@login_required
def view_invoice_pdf():
    # Получаем параметры, которые идентифицируют группу инвойса
    issued_to = request.args.get('issued_to')
    reference_job = request.args.get('reference_job')
    issued_by = request.args.get('issued_by')
    issue_date_str = request.args.get('issue_date')

    from datetime import datetime

    # Попытка распарсить дату
    try:
        issue_date = datetime.strptime(issue_date_str, '%Y-%m-%d %H:%M:%S.%f')
    except ValueError:
        try:
            issue_date = datetime.strptime(issue_date_str, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            issue_date = datetime.strptime(issue_date_str, '%Y-%m-%d')

    # Получаем все записи этой накладной
    records = IssuedPartRecord.query.filter(
        IssuedPartRecord.issued_to == issued_to,
        IssuedPartRecord.reference_job == reference_job,
        IssuedPartRecord.issued_by == issued_by,
        IssuedPartRecord.issue_date.between(
            datetime.combine(issue_date.date(), datetime.min.time()),
            datetime.combine(issue_date.date(), datetime.max.time())
        )
    ).all()

    from utils.invoice_generator import generate_view_pdf

    pdf_data = generate_view_pdf(records)  # Передаём список записей

    from io import BytesIO
    from flask import send_file

    return send_file(BytesIO(pdf_data),
                     as_attachment=True,
                     download_name=f"INVOICE_{issued_to}_{issue_date.strftime('%Y%m%d')}.pdf",
                     mimetype="application/pdf")



# Обновление отдельной позиции
@inventory_bp.route('/reports/update/<int:record_id>', methods=['POST'])
@login_required
def update_report_record(record_id):
    if current_user.role not in ['superadmin', 'admin']:
        flash("Access denied", "danger")
        return redirect(url_for('inventory.reports'))

    record = IssuedPartRecord.query.get_or_404(record_id)

    issued_to = request.form.get('issued_to', '').strip()
    reference_job = request.form.get('reference_job', '').strip()
    unit_cost_str = request.form.get('unit_cost', '').strip()
    issue_date_str = request.form.get('issue_date', '').strip()

    if not issued_to:
        flash("Issued To field cannot be empty.", "danger")
        return redirect(url_for('inventory.reports'))

    try:
        unit_cost = float(unit_cost_str)
        if unit_cost < 0:
            raise ValueError()
    except ValueError:
        flash("Invalid Unit Cost value.", "danger")
        return redirect(url_for('inventory.reports'))

    # Сохраняем дату только для superadmin
    if issue_date_str and current_user.role == 'superadmin':
        try:
            new_issue_date = datetime.strptime(issue_date_str, '%Y-%m-%d')
            record.issue_date = new_issue_date
        except ValueError:
            flash("Invalid Issue Date format.", "danger")
            return redirect(url_for('inventory.reports'))

    record.issued_to = issued_to
    record.reference_job = reference_job if reference_job else None
    record.unit_cost_at_issue = unit_cost

    db.session.commit()
    flash("Issued record updated successfully.", "success")
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


# Обновление данных накладной (issued_to, reference_job) для всей группы позиций
@inventory_bp.route('/reports/update_invoice', methods=['POST'])
@login_required
def update_invoice():
    if current_user.role not in ['superadmin', 'admin']:
        flash("Access denied", "danger")
        return redirect(url_for('inventory.reports'))

    action = request.form.get('action')
    issued_to_old = request.args.get('issued_to')
    reference_job_old = request.args.get('reference_job')
    issued_by = request.args.get('issued_by')
    issue_date_str = request.args.get('issue_date')

    # Попытка распарсить дату с разными форматами
    try:
        issue_date = datetime.strptime(issue_date_str, '%Y-%m-%d %H:%M:%S.%f')
    except ValueError:
        try:
            issue_date = datetime.strptime(issue_date_str, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            issue_date = datetime.strptime(issue_date_str, '%Y-%m-%d')

    if action == "save":
        new_issued_to = request.form.get('issued_to', '').strip()
        new_reference_job = request.form.get('reference_job', '').strip()

        if not new_issued_to:
            flash("Issued To field cannot be empty.", "danger")
            return redirect(url_for('inventory.reports'))

        records = IssuedPartRecord.query.filter(
            IssuedPartRecord.issued_to == issued_to_old,
            IssuedPartRecord.reference_job == reference_job_old,
            IssuedPartRecord.issued_by == issued_by,
            IssuedPartRecord.issue_date.between(
                datetime.combine(issue_date.date(), datetime.min.time()),
                datetime.combine(issue_date.date(), datetime.max.time())
            )
        ).all()

        for r in records:
            r.issued_to = new_issued_to
            r.reference_job = new_reference_job if new_reference_job else None

        db.session.commit()
        flash("Invoice updated successfully.", "success")
        return redirect(url_for('inventory.reports'))

    elif action == "cancel":
        records = IssuedPartRecord.query.filter(
            IssuedPartRecord.issued_to == issued_to_old,
            IssuedPartRecord.reference_job == reference_job_old,
            IssuedPartRecord.issued_by == issued_by,
            IssuedPartRecord.issue_date.between(
                datetime.combine(issue_date.date(), datetime.min.time()),
                datetime.combine(issue_date.date(), datetime.max.time())
            )
        ).all()

        for r in records:
            part = Part.query.get(r.part_id)
            if part:
                part.quantity += r.quantity
            db.session.delete(r)

        db.session.commit()
        flash("Invoice canceled and stock restored.", "success")
        return redirect(url_for('inventory.reports'))

    else:
        flash("Invalid action.", "danger")
        return redirect(url_for('inventory.reports'))


# Отмена всей накладной (группы записей)
@inventory_bp.route('/reports/cancel_invoice', methods=['POST'])
@login_required
def cancel_invoice():
    ALLOWED_CANCEL_ROLES = ('superadmin',)  # потом будет ('superadmin', 'admin')
    if current_user.role not in ALLOWED_CANCEL_ROLES:
        flash("Access denied (superadmin only).", "danger")
        return redirect(url_for('inventory.reports'))

    issued_to = request.form.get('issued_to')
    reference_job = request.form.get('reference_job')
    issued_by = request.form.get('issued_by')
    issue_date_str = request.form.get('issue_date')

    try:
        issue_date = datetime.strptime(issue_date_str, '%Y-%m-%d %H:%M:%S.%f')
    except ValueError:
        try:
            issue_date = datetime.strptime(issue_date_str, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            issue_date = datetime.strptime(issue_date_str, '%Y-%m-%d')

    records = IssuedPartRecord.query.filter(
        IssuedPartRecord.issued_to == issued_to,
        IssuedPartRecord.reference_job == reference_job,
        IssuedPartRecord.issued_by == issued_by,
        IssuedPartRecord.issue_date.between(
            datetime.combine(issue_date.date(), datetime.min.time()),
            datetime.combine(issue_date.date(), datetime.max.time())
        )
    ).all()

    for r in records:
        part = Part.query.get(r.part_id)
        if part:
            part.quantity += r.quantity
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

@inventory_bp.post('/reports/create_return_invoice')
@login_required
def create_return_invoice():
    issued_to      = request.form.get('issued_to', '')
    reference_job  = request.form.get('reference_job') or None
    issued_by      = request.form.get('issued_by', '')
    issue_date_str = request.form.get('issue_date', '')

    # 1) разбор даты
    from datetime import datetime
    try:
        issue_date = datetime.strptime(issue_date_str, '%Y-%m-%d %H:%M:%S.%f')
    except ValueError:
        try:
            issue_date = datetime.strptime(issue_date_str, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            issue_date = datetime.strptime(issue_date_str, '%Y-%m-%d')

    # 2) исходная группа накладной
    src_records = _fetch_invoice_group(issued_to, reference_job, issued_by, issue_date)
    if not src_records:
        flash("Source invoice not found.", "danger")
        return redirect(url_for('inventory.reports'))

    # защита от возврата-из-возврата
    if _is_return_group(src_records):
        flash("This invoice looks like a RETURN already.", "warning")
        return redirect(url_for('inventory.reports'))

    # 3) выбранные строки (чекбоксы)
    raw_ids = request.form.getlist('line_ids')
    if not raw_ids:
        flash("Select at least one line to return.", "warning")
        return redirect(
            url_for('inventory.reports_grouped') + "?" + urlencode({
                "start_date": issue_date.date().isoformat(),
                "end_date":   issue_date.date().isoformat(),
                "reference_job": reference_job or ""
            })
        )

    try:
        selected_ids = {int(x) for x in raw_ids}
    except ValueError:
        flash("Incorrect selection.", "danger")
        return redirect(url_for('inventory.reports'))

    # быстрый доступ по id из исходной группы
    by_id = {r.id: r for r in src_records if r.id is not None}

    # 4) reference для возвратной накладной
    ret_ref = f"RETURN #{issue_date.strftime('%Y%m%d')}-{(issued_to or '').strip().upper()}"

    created = 0
    from flask_login import current_user
    from datetime import datetime as _dt

    for lid in selected_ids:
        r = by_id.get(lid)
        if not r:
            continue  # чужая строка — игнор

        # --- доступный остаток к возврату (универсально) ---
        issued_qty   = int(r.quantity or 0)
        returned_qty = int(getattr(r, "qty_returned", 0) or 0)  # если поля нет — будет 0
        max_return   = max(issued_qty - returned_qty, 0)
        if max_return <= 0:
            continue

        # --- желаемое количество с формы ---
        form_qty = request.form.get(f'qty_{lid}', '0')
        try:
            req_qty = int(form_qty)
        except ValueError:
            req_qty = 0
        if req_qty <= 0:
            continue

        return_qty = min(req_qty, max_return)

        # --- создаём возвратную запись с отрицательным количеством ---
        part = Part.query.get(r.part_id)
        unit_cost = r.unit_cost_at_issue or (part.unit_cost if part else 0)

        ret = IssuedPartRecord(
            part_id=r.part_id,
            quantity=-return_qty,                      # выборочное кол-во с минусом
            issued_to=r.issued_to,
            reference_job=ret_ref,
            issued_by=current_user.username,
            issue_date=_dt.utcnow(),
            unit_cost_at_issue=unit_cost
        )

        # если у модели есть поле ссылки на исходную строку — заполним
        if hasattr(ret, "parent_line_id"):
            setattr(ret, "parent_line_id", r.id)

        db.session.add(ret)

        # --- вернуть на склад ---
        if part:
            part.quantity = (part.quantity or 0) + return_qty

        # --- аккумулируем частичный возврат, если поле существует ---
        if hasattr(r, "qty_returned"):
            r.qty_returned = (getattr(r, "qty_returned") or 0) + return_qty

        created += 1

    if created == 0:
        db.session.rollback()
        flash("Nothing to return for selected lines.", "warning")
        return redirect(url_for('inventory.reports'))

    db.session.commit()
    flash(f"Return invoice created: {created} item(s).", "success")

    # 5) показать созданный возврат в отчёте за сегодня
    params = {
        "start_date": _dt.utcnow().date().isoformat(),
        "end_date":   _dt.utcnow().date().isoformat(),
        "reference_job": ret_ref
    }
    return redirect(url_for('inventory.reports_grouped') + "?" + urlencode(params), code=303)


@inventory_bp.post('/reports/delete_return_invoice')
@login_required
def delete_return_invoice():
    if current_user.role != 'superadmin':
        flash("Only superadmin can delete return invoices.", "danger")
        return redirect(url_for('inventory.reports'))

    issued_to     = request.form.get('issued_to', '')
    reference_job = request.form.get('reference_job') or None
    issued_by     = request.form.get('issued_by', '')
    issue_date_str= request.form.get('issue_date', '')

    try:
        issue_date = datetime.strptime(issue_date_str, '%Y-%m-%d %H:%M:%S.%f')
    except ValueError:
        try:
            issue_date = datetime.strptime(issue_date_str, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            issue_date = datetime.strptime(issue_date_str, '%Y-%m-%d')

    records = _fetch_invoice_group(issued_to, reference_job, issued_by, issue_date)
    if not records:
        flash("Return invoice not found.", "danger")
        return redirect(url_for('inventory.reports'))

    if not _is_return_group(records):
        flash("Selected invoice is not a RETURN invoice.", "warning")
        return redirect(url_for('inventory.reports'))

    # 1) Откатываем склад: при возврате мы прибавляли abs(qty) → сейчас уменьшаем на abs(qty)
    for r in records:
        part = Part.query.get(r.part_id)
        if part and (r.quantity or 0) < 0:
            # уменьшить на abs(qty); защита от отрицательных остатков
            new_qty = (part.quantity or 0) - abs(r.quantity)
            part.quantity = max(new_qty, 0)

    # 2) Удаляем сами записи возврата
    for r in records:
        db.session.delete(r)

    db.session.commit()
    flash("Return invoice deleted and stock adjusted.", "success")
    return redirect(url_for('inventory.reports'))












