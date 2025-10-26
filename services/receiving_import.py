# services/receiving_import.py
from datetime import datetime, date
from extensions import db
from models import GoodsReceipt, GoodsReceiptLine
from services.receiving import post_receiving_batch  # твой сервис постинга

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
) -> GoodsReceipt:

    """
    Создаёт GoodsReceipt + GoodsReceiptLine.
    Если auto_post=True:
      - после создания сразу делает постинг на склад (post_receiving_batch),
      - гарантирует, что статус в базе = 'posted',
      - возвращает уже свежий объект.
    Очень важно: снаружи НЕ надо потом снова коммитить этот объект,
    иначе ты перезапишешь статус назад в 'draft'.
    """

    # 1. создаём draft GR
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

    # этот commit фиксирует строки и draft-состояние в БД
    db.session.commit()

    if auto_post:
        # 2. постим (эта функция:
        #    - добавит qty в склад
        #    - проставит status='posted', posted_at, posted_by
        #    - сделает commit()
        post_receiving_batch(gr.id, current_user_id=created_by)

        # 3. очень важно: берём ЧИСТУЮ копию из базы
        gr = db.session.get(GoodsReceipt, gr.id)

        # 4. разрываем связь с сессией, чтобы никто потом не мог снова "случайно" записать draft
        db.session.expunge(gr)

    else:
        # если не постим сразу — оставляем как draft, но тоже expunge,
        # чтобы наружу ушёл "слепок", а не живой объект, который потом
        # может случайно перезаписать состояние
        db.session.expunge(gr)

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
