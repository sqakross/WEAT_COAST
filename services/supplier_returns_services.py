# services/supplier_returns_services.py
from __future__ import annotations
from datetime import datetime
from typing import Dict, Any

from extensions import db
from models import SupplierReturnBatch, SupplierReturnItem, Part, GoodsReceiptLine

from services.lot_costing import pick_receipt_line_for_return, receipt_line_base_cost


class SupplierReturnError(Exception):
    pass


def _find_part(pn: str) -> Part | None:
    if not pn:
        return None
    return Part.query.filter(Part.part_number == pn).first()


def _get_return_invoice_ref(it: SupplierReturnItem) -> str | None:
    """
    Try to read invoice/ref from the return item (supports different field names).
    Adjust if you have a single known field name.
    """
    for attr in ("inv_ref", "invoice_number", "receipt_invoice", "invoice_ref"):
        v = getattr(it, attr, None)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return None


def recalc_batch_totals(batch: SupplierReturnBatch) -> Dict[str, Any]:
    """
    NEW RULE:
      - RETURN cost is ALWAYS BASE cost (from receipt lot), never Part.unit_cost (which may include expenses).
      - If unit_cost is empty/0: we MUST resolve cost from receipt lot:
          A) source_receipt_line_id (best)
          B) strict invoice match: invoice + part_number (NO latest fallback)
          C) if cannot resolve -> error (no silent fallback)
    """
    errors: Dict[int, str] = {}
    total_items = 0
    total_value = 0.0

    for idx, it in enumerate(batch.items or []):
        pn = (it.part_number or "").strip()
        if not pn:
            errors[idx] = "Part number required."
            it.total_cost = 0.0
            continue

        p = _find_part(pn)
        if not p:
            errors[idx] = f"Part '{pn}' not found in inventory."
            q = max(0, int(it.qty_returned or 0))
            c = float(it.unit_cost or 0.0)
            it.total_cost = round(q * c, 2)
            total_items += q
            total_value += it.total_cost
            continue

        # fill name/location (safe)
        if not it.part_name:
            it.part_name = p.name or ""

        loc = (it.location or "").strip().lower()
        if not loc or loc == "auto":
            it.location = p.location or ""

        # normalize qty
        q = max(0, int(it.qty_returned or 0))
        it.qty_returned = q

        # ===== COST RESOLUTION (BASE ONLY) =====
        # If user explicitly set unit_cost > 0 — keep it (manual override).
        need_cost = (it.unit_cost is None) or (float(it.unit_cost) <= 0.0)

        if need_cost:
            # A) if item has source_receipt_line_id, use that lot base
            src_line_id = getattr(it, "source_receipt_line_id", None)
            if src_line_id:
                line = db.session.get(GoodsReceiptLine, int(src_line_id))
                if not line:
                    errors[idx] = f"Receipt lot not found (source_receipt_line_id={src_line_id}) for part {pn}."
                    it.unit_cost = 0.0
                    it.total_cost = 0.0
                    continue
                it.unit_cost = receipt_line_base_cost(line)
            else:
                # B) strict invoice match required
                inv = _get_return_invoice_ref(it)
                if not inv:
                    errors[idx] = (
                        f"Invoice required to calculate BASE return cost for {pn}. "
                        f"Please enter receipt invoice (or link to receipt lot)."
                    )
                    it.unit_cost = 0.0
                    it.total_cost = 0.0
                    continue

                line, src = pick_receipt_line_for_return(part_number=pn, inv_ref=inv)
                if not line:
                    errors[idx] = (
                        f"Receipt line not found for {pn} with invoice '{inv}' (strict match). "
                        f"Cannot calculate BASE return cost."
                    )
                    it.unit_cost = 0.0
                    it.total_cost = 0.0
                    continue

                it.unit_cost = receipt_line_base_cost(line)

        # finalize line totals
        c = float(it.unit_cost or 0.0)
        it.unit_cost = c
        it.total_cost = round(q * c, 2)

        total_items += q
        total_value += it.total_cost

    batch.total_items = int(total_items)
    batch.total_value = float(round(total_value, 2))

    return {"ok": len(errors) == 0, "errors": errors}

def post_batch(batch_id: int, actor: str | None = None) -> Dict[str, Any]:
    """
    Постинг возврата:
      - валидация и пересчёт;
      - уменьшение склада (Part.quantity -= qty_returned);
      - проставление статуса posted/posted_at/by.
    """
    b = SupplierReturnBatch.query.get(batch_id)
    if not b:
        raise SupplierReturnError("Batch not found.")

    if (b.status or "draft") == "posted":
        return {"ok": True, "already": True}

    # пересчёт + валидация
    info = recalc_batch_totals(b)
    if not info.get("ok"):
        # не даём постить, пока не исправят строки
        db.session.flush()
        return {"ok": False, "errors": info["errors"]}

    # проверка наличия/остатков и уменьшение
    per_row_errors: Dict[int, str] = {}
    for idx, it in enumerate(b.items or []):
        p = _find_part(it.part_number)
        if not p:
            per_row_errors[idx] = f"Part '{it.part_number}' not found."
            continue
        q = int(it.qty_returned or 0)
        if q <= 0:
            continue
        if int(p.quantity or 0) < q:
            per_row_errors[idx] = f"Not enough stock for {it.part_number} (have {p.quantity}, need {q})."
            continue
        # уменьшаем склад
        p.quantity = int(p.quantity or 0) - q
        db.session.add(p)

    if per_row_errors:
        db.session.flush()
        return {"ok": False, "errors": per_row_errors}

    # ok → ставим статус
    b.status = "posted"
    b.posted_at = datetime.utcnow()
    b.posted_by = (actor or "")[:120] if actor else None
    db.session.add(b)
    db.session.commit()

    return {"ok": True}


def unpost_batch(batch_id: int, actor: str | None = None) -> Dict[str, Any]:
    """
    Откат постинга:
      - возвращаем количество на склад (Part.quantity += qty_returned)
      - статус -> draft, чистим метаданные постинга
    """
    b = SupplierReturnBatch.query.get(batch_id)
    if not b:
        raise SupplierReturnError("Batch not found.")

    if (b.status or "draft") != "posted":
        return {"ok": False, "errors": {"_": "Only posted batch can be unposted."}}

    per_row_errors: Dict[int, str] = {}
    for idx, it in enumerate(b.items or []):
        p = _find_part(it.part_number)
        if not p:
            per_row_errors[idx] = f"Part '{it.part_number}' disappeared from inventory."
            continue
        q = int(it.qty_returned or 0)
        if q <= 0:
            continue
        p.quantity = int(p.quantity or 0) + q
        db.session.add(p)

    if per_row_errors:
        db.session.flush()
        return {"ok": False, "errors": per_row_errors}

    b.status = "draft"
    b.posted_at = None
    b.posted_by = None
    db.session.add(b)
    db.session.commit()

    return {"ok": True}
