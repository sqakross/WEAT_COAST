# services/lot_costing.py
from __future__ import annotations

from typing import Optional, Tuple
from sqlalchemy import func

from extensions import db
from models import GoodsReceipt, GoodsReceiptLine, ReceivingBatch, ReceivingItem

def _norm_inv(s: str) -> str:
    s = (s or "").strip()
    # normalize leading zeros (e.g. "0002066" vs "2066")
    s2 = s.lstrip("0")
    return s2 if s2 else s

def pick_receipt_line_for_return(*, part_number: str, inv_ref: str | None):
    pn = (part_number or "").strip().upper()
    inv = _norm_inv(inv_ref or "")

    if not pn:
        return None, "missing_part_number"
    if not inv:
        return None, "missing_invoice"

    q1 = (
        db.session.query(GoodsReceiptLine)
        .join(GoodsReceipt, GoodsReceiptLine.goods_receipt_id == GoodsReceipt.id)
        .filter(func.upper(GoodsReceiptLine.part_number) == pn)
        .filter(func.lower(func.coalesce(GoodsReceipt.status, "")) == "posted")
        .filter(func.ltrim(func.coalesce(GoodsReceipt.invoice_number, ""), "0") == inv)
        .order_by(
            GoodsReceipt.posted_at.desc().nullslast(),
            GoodsReceipt.id.desc(),
            GoodsReceiptLine.id.desc(),
        )
    )
    line = q1.first()
    if line is not None:
        return line, "goods_receipt_match"

    return None, "receipt_inv_not_found"

def receipt_line_base_cost(line: "GoodsReceiptLine") -> float:
    """
    RETURN cost (BASE ONLY, no extras).
    Priority:
      1) explicit base fields
      2) derive base = actual_unit_cost - extra_alloc_per_unit
      3) if actual exists and equals unit_cost -> do NOT return unit_cost (it's adjusted) => 0.0
      4) last resort: unit_cost (for very old lots where it was base)
    """
    # helper: safe float
    def _f(x):
        try:
            if x is None:
                return None
            return float(x)
        except Exception:
            return None

    # 1) explicit base fields (your model uses base_unit_cost)
    for attr in ("base_unit_cost", "unit_cost_base", "base_cost"):
        v = _f(getattr(line, attr, None))
        if v is not None and v > 0:
            return round(v, 4)

    # 2) derive from actual - alloc
    actual = _f(getattr(line, "actual_unit_cost", None))
    alloc  = _f(getattr(line, "extra_alloc_per_unit", None))
    if actual is not None and alloc is not None:
        base = actual - alloc
        if base < 0:
            base = 0.0
        return round(base, 4)

    # 3) if actual exists and unit_cost matches it -> unit_cost is adjusted => return 0.0
    unit_cost = _f(getattr(line, "unit_cost", None)) or 0.0
    if actual is not None and abs(actual - unit_cost) < 0.0001:
        return 0.0

    # 4) fallback (old data)
    return round(unit_cost, 4)


def receipt_line_cost(line: GoodsReceiptLine) -> float:
    """
    Issue cost:
      - prefer actual_unit_cost (includes expenses/alloc if your receiving set it)
      - else unit_cost
    This function MUST exist because issue code imports it.
    """
    try:
        if getattr(line, "actual_unit_cost", None) is not None:
            return float(line.actual_unit_cost or 0.0)
    except Exception:
        pass

    try:
        return float(getattr(line, "unit_cost", 0.0) or 0.0)
    except Exception:
        return 0.0

# -------------------------------------------------------------------
# Backward-compat: older code imports this name
# -------------------------------------------------------------------
def pick_receipt_line_for_issue(part_id: int, qty_needed: int = 1):
    """
    Pick a GoodsReceiptLine / ReceivingItem line to issue stock from.
    FIFO by created/received date.

    Returns a model instance (receipt line) or None.

    This function exists for backward compatibility because some routes
    import `pick_receipt_line_for_issue` from services.lot_costing.
    """
    try:
        from models import GoodsReceiptLine  # adjust if your model name differs
    except Exception:
        try:
            # some projects call it ReceivingItem / ReceivingLine
            from models import ReceivingItem as GoodsReceiptLine  # type: ignore
        except Exception:
            try:
                from models import ReceivingLine as GoodsReceiptLine  # type: ignore
            except Exception:
                return None

    # IMPORTANT: detect columns safely (project variants)
    def _has(attr: str) -> bool:
        return hasattr(GoodsReceiptLine, attr)

    part_field = "part_id" if _has("part_id") else None
    qty_field = "qty" if _has("qty") else ("quantity" if _has("quantity") else None)

    # remaining/available quantity fields differ across versions
    rem_field = None
    for cand in ("qty_remaining", "remaining_qty", "remaining", "qty_left"):
        if _has(cand):
            rem_field = cand
            break

    # date ordering fields
    order_field = None
    for cand in ("received_at", "created_at", "invoice_date", "date"):
        if _has(cand):
            order_field = cand
            break

    if not part_field or not qty_field:
        return None

    q = GoodsReceiptLine.query.filter(getattr(GoodsReceiptLine, part_field) == int(part_id))

    # only lines with something available
    if rem_field:
        q = q.filter(getattr(GoodsReceiptLine, rem_field) > 0)
    else:
        # fallback: if no remaining field exists, just require qty > 0
        q = q.filter(getattr(GoodsReceiptLine, qty_field) > 0)

    if order_field:
        q = q.order_by(getattr(GoodsReceiptLine, order_field).asc())

    return q.first()