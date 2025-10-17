# services/receiving_import.py
from datetime import datetime, date
from extensions import db
from models import GoodsReceipt, GoodsReceiptLine
from services.receiving import post_receiving_batch  # твой сервис постинга

def create_receiving_from_rows(*, supplier_name: str, invoice_number: str | None,
                               invoice_date: date | None, currency: str | None,
                               notes: str | None, rows: list[dict], created_by=None,
                               auto_post: bool = False) -> GoodsReceipt:
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
    db.session.flush()

    line_no = 1
    for r in rows:
        pn = (r.get("part_number") or r.get("pn") or "").strip()
        if not pn:
            continue
        qty = int((r.get("quantity") or r.get("qty") or 0) or 0)
        if qty <= 0:
            continue
        line = GoodsReceiptLine(
            goods_receipt_id=gr.id,
            line_no=line_no,
            part_number=pn,
            part_name=(r.get("part_name") or r.get("description") or r.get("descr") or "").strip() or None,
            quantity=qty,
            unit_cost=float((r.get("unit_cost") or r.get("price") or 0) or 0),
            location=(r.get("location") or r.get("supplier") or "").strip() or None,
        )
        db.session.add(line)
        line_no += 1

    db.session.commit()

    if auto_post:
        # импорт внутри функции, чтобы избежать возможного циклического импорта
        post_receiving_batch(gr.id, current_user_id=created_by)

    return gr

# services/receiving_import.py
def _coalesce_same_parts(rows: list[dict]) -> list[dict]:
    acc = {}
    for r in rows:
        pn = (r.get("part_number") or r.get("pn") or "").strip().upper()
        if not pn:
            continue
        key = (pn, (r.get("unit_cost") or 0))
        if key not in acc:
            acc[key] = {
                "part_number": pn,
                "part_name": (r.get("part_name") or r.get("description") or r.get("descr") or "").strip(),
                "quantity": int((r.get("quantity") or r.get("qty") or 0) or 0),
                "unit_cost": float((r.get("unit_cost") or r.get("price") or 0) or 0),
                "location": (r.get("location") or r.get("supplier") or "").strip(),
            }
        else:
            acc[key]["quantity"] += int((r.get("quantity") or r.get("qty") or 0) or 0)
    return [v for v in acc.values() if v["quantity"] > 0]
