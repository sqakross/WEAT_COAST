# services/lot_costing.py
from __future__ import annotations

from sqlalchemy import func
import re
from extensions import db
from models import GoodsReceipt, GoodsReceiptLine, ReceivingBatch, ReceivingItem


def _norm_inv(s: str) -> str:
    s = (s or "").strip()
    s2 = s.lstrip("0")
    return s2 if s2 else s

def _split_invoice_refs(raw: str | None) -> list[str]:
    """
    Accepts strings like:
      "71619166 91852629"
      "71619166,91852629"
      "71619166/91852629"
      "71619166; 91852629"
    Returns normalized invoice tokens without leading zeros.
    """
    s = (raw or "").strip()
    if not s:
        return []
    # split by common separators: space, comma, semicolon, slash, pipe, newline, tab
    parts = re.split(r"[,\s;/|]+", s)
    out: list[str] = []
    for p in parts:
        p = _norm_inv(p)
        if p:
            out.append(p)
    # de-dupe but keep order
    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq

def _is_posted(status_col):
    return func.lower(func.coalesce(status_col, "")) == "posted"

def pick_receipt_line_for_return(*, part_number: str, inv_ref: str | None):
    pn = (part_number or "").strip().upper()
    invs = _split_invoice_refs(inv_ref)

    if not pn:
        return None, "missing_part_number"

    # ============================================================
    # A) STRICT invoice match
    #    If source row has invoice ref -> must match that invoice only
    # ============================================================
    if invs:
        for inv in invs:
            q1 = (
                db.session.query(GoodsReceiptLine)
                .join(GoodsReceipt, GoodsReceiptLine.goods_receipt_id == GoodsReceipt.id)
                .filter(func.upper(GoodsReceiptLine.part_number) == pn)
                .filter(_is_posted(GoodsReceipt.status))
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

    # ============================================================
    # B) STOCK mode
    #    No invoice on source row -> return back to stock by part_number only
    # ============================================================
    q2 = (
        db.session.query(GoodsReceiptLine)
        .join(GoodsReceipt, GoodsReceiptLine.goods_receipt_id == GoodsReceipt.id)
        .filter(func.upper(GoodsReceiptLine.part_number) == pn)
        .filter(_is_posted(GoodsReceipt.status))
        .order_by(
            GoodsReceipt.posted_at.desc().nullslast(),
            GoodsReceipt.id.desc(),
            GoodsReceiptLine.id.desc(),
        )
    )

    line = q2.first()
    if line is not None:
        return line, "stock_no_invoice"

    return None, "no_stock_receipt_found"

def receipt_line_base_cost(line: "GoodsReceiptLine") -> float:
    def _f(x):
        try:
            if x is None:
                return None
            return float(x)
        except Exception:
            return None

    for attr in ("base_unit_cost", "unit_cost_base", "base_cost"):
        v = _f(getattr(line, attr, None))
        if v is not None and v > 0:
            return round(v, 4)

    actual = _f(getattr(line, "actual_unit_cost", None))
    alloc = _f(getattr(line, "extra_alloc_per_unit", None))
    if actual is not None and alloc is not None:
        base = actual - alloc
        if base < 0:
            base = 0.0
        return round(base, 4)

    unit_cost = _f(getattr(line, "unit_cost", None)) or 0.0
    if actual is not None and abs(actual - unit_cost) < 0.0001:
        return 0.0

    return round(unit_cost, 4)


def receipt_line_cost(line) -> float:
    """
    Issue cost:
      - prefer actual_unit_cost
      - else unit_cost
    Works for GoodsReceiptLine and (optionally) ReceivingItem if it has those fields.
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


def pick_receipt_line_for_issue(
    part_id: int | None = None,
    qty_needed: int = 1,
    part_number: str | None = None,
    inv_ref: str | None = None,
    prefer_latest: bool = False,  # ✅ NEW (default keeps old behavior)
):
    """
    Picker for issuing stock.

    Rules:
      - if inv_ref provided -> STRICT match by POSTED invoice (latest within that invoice)
      - if inv_ref is None:
          - prefer_latest=False -> FIFO (old behavior)
          - prefer_latest=True  -> LATEST (newest posted lot)

    Return style:
      - if called with part_number/inv_ref -> returns (line, src)
      - else -> returns line
    """
    want_tuple = (part_number is not None) or (inv_ref is not None)

    if not part_number and part_id is not None:
        try:
            from models import Part
            p = Part.query.get(int(part_id))
            if not p:
                return (None, "part_not_found") if want_tuple else None
            part_number = p.part_number
        except Exception:
            return (None, "part_lookup_failed") if want_tuple else None

    pn = (part_number or "").strip().upper()
    if not pn:
        return (None, "missing_part_number") if want_tuple else None

    # optional "remaining" field on GoodsReceiptLine
    rem_field = None
    for cand in ("qty_remaining", "remaining_qty", "remaining", "qty_left"):
        if hasattr(GoodsReceiptLine, cand):
            rem_field = cand
            break

    inv = _norm_inv(inv_ref or "") if inv_ref else ""

    # ------------------------------------------------------------
    # fallback remaining calc when no rem_field exists:
    # remaining = GoodsReceiptLine.quantity - SUM(IssuedPartRecord.quantity where source_receipt_line_id=line.id)
    # ------------------------------------------------------------
    IssuedPartRecord = None
    issued_sum = None
    try:
        from models import IssuedPartRecord as _IPR
        IssuedPartRecord = _IPR
        issued_sum = func.coalesce(func.sum(IssuedPartRecord.quantity), 0)
    except Exception:
        IssuedPartRecord = None
        issued_sum = None

    def _apply_remaining_filter_goodsreceipt(q):
        if rem_field:
            return q.filter(getattr(GoodsReceiptLine, rem_field) > 0)

        if (
            IssuedPartRecord is not None
            and issued_sum is not None
            and hasattr(GoodsReceiptLine, "quantity")
            and hasattr(IssuedPartRecord, "source_receipt_line_id")
        ):
            q = q.outerjoin(
                IssuedPartRecord,
                IssuedPartRecord.source_receipt_line_id == GoodsReceiptLine.id,
            )
            q = q.group_by(GoodsReceiptLine.id, GoodsReceipt.id)
            q = q.having((func.coalesce(GoodsReceiptLine.quantity, 0) - issued_sum) > 0)
            return q

        return q

    # ============================================================
    # A) STRICT invoice match
    # ============================================================
    if inv:
        qg = (
            db.session.query(GoodsReceiptLine)
            .join(GoodsReceipt, GoodsReceiptLine.goods_receipt_id == GoodsReceipt.id)
            .filter(func.upper(GoodsReceiptLine.part_number) == pn)
            .filter(_is_posted(GoodsReceipt.status))
            .filter(func.ltrim(func.coalesce(GoodsReceipt.invoice_number, ""), "0") == inv)
        )

        qg = _apply_remaining_filter_goodsreceipt(qg)

        # within invoice: pick newest posted line
        qg = qg.order_by(
            GoodsReceipt.posted_at.desc().nullslast(),
            GoodsReceipt.id.desc(),
            GoodsReceiptLine.id.desc(),
        )

        line = qg.first()
        if line is not None:
            return (line, "receipt_inv_match") if want_tuple else line

        # ReceivingBatch fallback (only if schema supports it)
        fk = None
        if hasattr(ReceivingItem, "receiving_batch_id"):
            fk = ReceivingItem.receiving_batch_id
        elif hasattr(ReceivingItem, "batch_id"):
            fk = ReceivingItem.batch_id

        if fk is not None:
            qr = (
                db.session.query(ReceivingItem)
                .join(ReceivingBatch, fk == ReceivingBatch.id)
                .filter(func.upper(ReceivingItem.part_number) == pn)
                .filter(_is_posted(ReceivingBatch.status))
                .filter(func.ltrim(func.coalesce(ReceivingBatch.invoice_number, ""), "0") == inv)
            )

            rem2 = None
            for cand in ("qty_remaining", "remaining_qty", "remaining", "qty_left"):
                if hasattr(ReceivingItem, cand):
                    rem2 = cand
                    break
            if rem2:
                qr = qr.filter(getattr(ReceivingItem, rem2) > 0)

            if hasattr(ReceivingBatch, "posted_at"):
                qr = qr.order_by(
                    ReceivingBatch.posted_at.desc().nullslast(),
                    ReceivingBatch.id.desc(),
                    ReceivingItem.id.desc(),
                )
            else:
                qr = qr.order_by(ReceivingBatch.id.desc(), ReceivingItem.id.desc())

            line2 = qr.first()
            if line2 is not None:
                return (line2, "receipt_inv_match") if want_tuple else line2

        return (None, "receipt_inv_not_found") if want_tuple else None

    # ============================================================
    # B) STOCK mode (inv_ref=None):
    #    - prefer_latest=False -> FIFO (old)
    #    - prefer_latest=True  -> LATEST (new)
    # ============================================================
    q = (
        db.session.query(GoodsReceiptLine)
        .join(GoodsReceipt, GoodsReceiptLine.goods_receipt_id == GoodsReceipt.id)
        .filter(func.upper(GoodsReceiptLine.part_number) == pn)
        .filter(_is_posted(GoodsReceipt.status))
    )

    q = _apply_remaining_filter_goodsreceipt(q)

    if prefer_latest:
        q = q.order_by(
            GoodsReceipt.posted_at.desc().nullslast(),
            GoodsReceipt.id.desc(),
            GoodsReceiptLine.id.desc(),
        )
    else:
        q = q.order_by(
            GoodsReceipt.posted_at.asc().nullslast(),
            GoodsReceipt.id.asc(),
            GoodsReceiptLine.id.asc(),
        )

    line = q.first()

    if want_tuple:
        if not line:
            return None, "no_stock"
        return line, ("latest" if prefer_latest else "fifo")

    return line

