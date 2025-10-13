from datetime import datetime
from extensions import db
from sqlalchemy import func
from models import ReceivingBatch, ReceivingItem
from models import Part  # твоя модель деталей

def _get_part_on_hand(part):
    # у тебя либо Part.on_hand, либо Part.quantity
    if hasattr(part, "on_hand"):
        return int(part.on_hand or 0)
    return int(getattr(part, "quantity", 0) or 0)

def _set_part_on_hand(part, value: int):
    if hasattr(part, "on_hand"):
        part.on_hand = int(value)
    else:
        part.quantity = int(value)

def post_receiving_batch(batch_id: int, current_user_id: int | None = None):
    batch = ReceivingBatch.query.get(batch_id)
    if not batch:
        raise ValueError("Receiving batch not found")

    if batch.status == "posted":
        return batch  # уже применили

    for it in batch.items:
        pn = (it.part_number or "").strip().upper()
        qty = int(it.quantity or 0)
        if not pn or qty <= 0:
            continue

        # ищем Part по PN
        part = Part.query.filter(func.upper(Part.part_number) == pn).first()
        if not part:
            # создаём черновик детали, если нет
            kwargs = dict(
                part_number=pn,
                part_name=it.part_name or "",
            )
            # подстрой под свои названия полей (supplier, location, warehouse и т.д.)
            if hasattr(Part, "supplier"):
                kwargs["supplier"] = (batch.supplier_name or "")[:120]
            if hasattr(Part, "location"):
                kwargs["location"] = (it.location or "")[:64]
            if hasattr(Part, "unit_cost") and it.unit_cost is not None:
                kwargs["unit_cost"] = float(it.unit_cost or 0)

            part = Part(**kwargs)
            db.session.add(part)

        # увеличиваем остаток
        on_hand = _get_part_on_hand(part) + qty
        _set_part_on_hand(part, on_hand)

        # обновим last_cost / unit_cost при желании
        if hasattr(part, "last_cost") and it.unit_cost is not None:
            part.last_cost = float(it.unit_cost or 0)
        elif hasattr(part, "unit_cost") and it.unit_cost is not None:
            part.unit_cost = float(it.unit_cost or 0)

        # обновим location, если задана
        if hasattr(part, "location") and (it.location or "").strip():
            part.location = (it.location or "").strip()[:64]

    batch.status = "posted"
    batch.posted_at = datetime.utcnow()
    if current_user_id is not None:
        batch.posted_by = int(current_user_id)

    db.session.commit()
    return batch
