# services/receiving_import.py
from __future__ import annotations

from datetime import datetime, date
from extensions import db
from models import GoodsReceipt, GoodsReceiptLine
from services.receiving import post_receiving_batch


def _num_float(x, default=0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(str(x).strip())
    except Exception:
        return float(default)


def _num_int(x, default=0) -> int:
    try:
        if x is None:
            return int(default)
        return int(float(str(x).strip()))
    except Exception:
        return int(default)


def _coalesce_same_parts(rows: list[dict]) -> list[dict]:
    """
    Склеиваем одинаковые PN + unit_cost (как у тебя было),
    но нормализуем:
      - PN -> UPPER
      - unit_cost -> float
      - qty -> int
    """
    acc: dict[tuple[str, float], dict] = {}

    for r in (rows or []):
        pn = (r.get("part_number") or r.get("pn") or "").strip().upper()
        if not pn:
            continue

        qty = _num_int(r.get("quantity") or r.get("qty") or 0, 0)
        if qty <= 0:
            continue

        unit_cost = _num_float(r.get("unit_cost") or r.get("price") or 0, 0.0)

        key = (pn, unit_cost)
        if key not in acc:
            acc[key] = {
                "part_number": pn,
                "part_name": (r.get("part_name") or r.get("description") or r.get("descr") or "").strip(),
                "quantity": qty,
                "unit_cost": unit_cost,
                "location": (r.get("location") or r.get("supplier") or "").strip(),
            }
        else:
            acc[key]["quantity"] += qty

    return [v for v in acc.values() if int(v.get("quantity") or 0) > 0]


def create_receiving_from_rows(
    *,
    supplier_name: str,
    invoice_number: str | None,
    invoice_date: date | None,
    currency: str | None,
    notes: str | None,
    rows: list[dict],
    created_by=None,
    auto_post: bool = False,
) -> GoodsReceipt:
    """
    Создаёт GoodsReceipt + GoodsReceiptLine.

    ГЛАВНОЕ:
    - unit_cost: исходная цена строки (как в документе)
    - base_unit_cost: цена без extra (по умолчанию = unit_cost)
    - extra_alloc_per_unit: по умолчанию 0.0
    - actual_unit_cost: база + extra (по умолчанию = unit_cost)

    Если auto_post=True:
      - после создания сразу делает постинг на склад (post_receiving_batch),
      - возвращает свежий объект из БД.
    """

    gr = GoodsReceipt(
        supplier_name=(supplier_name or "").strip(),
        invoice_number=(invoice_number or "").strip() or None,
        invoice_date=invoice_date,
        currency=(currency or "USD").strip()[:8],
        notes=(notes or "").strip() or None,
        status="draft",
        created_at=datetime.utcnow(),
        created_by=created_by,
    )
    db.session.add(gr)
    db.session.flush()  # есть gr.id

    line_no = 1
    for r in (rows or []):
        pn = (r.get("part_number") or r.get("pn") or "").strip().upper()
        if not pn:
            continue

        qty = _num_int(r.get("quantity") or r.get("qty") or 0, 0)
        if qty <= 0:
            continue

        unit_cost = _num_float(r.get("unit_cost") or r.get("price") or 0, 0.0)

        # по умолчанию:
        base = unit_cost
        extra = 0.0
        actual = unit_cost  # base + extra

        line = GoodsReceiptLine(
            goods_receipt_id=gr.id,
            line_no=line_no,
            part_number=pn,
            part_name=(r.get("part_name") or r.get("description") or r.get("descr") or "").strip() or None,
            quantity=qty,
            unit_cost=unit_cost,
            location=(r.get("location") or r.get("supplier") or "").strip() or None,

            # ✅ new cost model fields
            base_unit_cost=base,
            extra_alloc_per_unit=extra,
            actual_unit_cost=actual,
        )
        db.session.add(line)
        line_no += 1

    # фиксируем строки и draft-состояние в БД
    db.session.commit()

    if auto_post:
        # постинг делает commit()
        post_receiving_batch(gr.id, current_user_id=created_by)

        # чистая копия из БД
        gr = db.session.get(GoodsReceipt, gr.id)
        db.session.expunge(gr)
    else:
        db.session.expunge(gr)

    return gr
