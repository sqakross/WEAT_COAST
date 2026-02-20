# services/lot_costing.py
from __future__ import annotations

from typing import Optional, Tuple
from sqlalchemy import func

from extensions import db
from models import GoodsReceipt, GoodsReceiptLine


def pick_receipt_line_for_issue(
    *,
    part_number: str,
    inv_ref: str | None = None,
) -> tuple[GoodsReceiptLine | None, str]:
    """
    Возвращает (GoodsReceiptLine | None, cost_source).

    Логика:
      1) Если inv_ref задан -> пытаемся найти строку прихода по invoice_number + part_number (case-insensitive).
         Если нашли -> cost_source='receipt_inv_match'
      2) Иначе берём самый свежий posted lot по part_number.
         -> cost_source='receipt_latest_fallback'
      3) Если не нашли вообще -> (None, 'fallback_part_unit_cost')

    "Самый свежий" = по (posted_at desc, goods_receipts.id desc, goods_receipt_lines.id desc)
    """
    pn = (part_number or "").strip().upper()
    inv = (inv_ref or "").strip()
    if not pn:
        return None, "fallback_part_unit_cost"

    base_q = (
        db.session.query(GoodsReceiptLine)
        .join(GoodsReceipt, GoodsReceiptLine.goods_receipt_id == GoodsReceipt.id)
        .filter(func.upper(GoodsReceiptLine.part_number) == pn)
        .filter(func.lower(func.coalesce(GoodsReceipt.status, "")) == "posted")
        .order_by(
            GoodsReceipt.posted_at.desc().nullslast(),
            GoodsReceipt.id.desc(),
            GoodsReceiptLine.id.desc(),
        )
    )

    # 1) Invoice match
    if inv:
        q_inv = base_q.filter(func.lower(func.coalesce(GoodsReceipt.invoice_number, "")) == inv.lower())
        line = q_inv.first()
        if line is not None:
            return line, "receipt_inv_match"

    # 2) Latest fallback
    line = base_q.first()
    if line is not None:
        return line, "receipt_latest_fallback"

    return None, "fallback_part_unit_cost"


def receipt_line_cost(line: GoodsReceiptLine) -> float:
    """
    Берём себестоимость для выдачи:
      - если actual_unit_cost заполнен -> он
      - иначе unit_cost
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
