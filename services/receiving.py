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


def _sanitize_part_kwargs(raw: dict) -> dict:
    """
    Приводит kwargs к фактическим колонкам Part:
    - мапит part_name -> name, если колонка name есть и name ещё не задано;
    - дублирует name из part_number, если name есть и пустое;
    - удаляет ключи, которых нет в Part.
    """
    allowed = {c.name for c in Part.__table__.columns}
    data = dict(raw or {})

    # part_name -> name (только если у Part есть 'name' и name не задан)
    if "part_name" in data and "name" in allowed and not data.get("name"):
        data["name"] = data.get("part_name")

    # если name есть как колонка и не задано — подставим PN
    if "name" in allowed and not data.get("name"):
        data["name"] = data.get("part_number") or ""

    # выкинуть неизвестные ключи
    data = {k: v for k, v in data.items() if k in allowed}
    return data

def post_receiving_batch(batch_id: int, current_user_id: int | None = None):
    batch = ReceivingBatch.query.get(batch_id)
    if not batch:
        raise ValueError("Receiving batch not found")

    if (batch.status or "").lower() == "posted":
        return batch  # уже применили

    for it in (batch.items or []):  # alias items -> lines
        pn = (it.part_number or "").strip().upper()
        qty = int(it.quantity or 0)
        if not pn or qty <= 0:
            continue

        # ищем Part по PN (без учёта регистра)
        part = Part.query.filter(func.upper(Part.part_number) == pn).first()

        if not part:
            # создать Part (как раньше)
            raw_kwargs = {
                "part_number": pn,
                "part_name": it.part_name or "",  # _sanitize_part_kwargs смапит на name, если есть
            }
            if hasattr(Part, "supplier"):
                raw_kwargs["supplier"] = (batch.supplier_name or "")[:120]
            if hasattr(Part, "location"):
                raw_kwargs["location"] = (it.location or "")[:64]
            if hasattr(Part, "unit_cost") and it.unit_cost is not None:
                raw_kwargs["unit_cost"] = float(it.unit_cost or 0)

            kwargs = _sanitize_part_kwargs(raw_kwargs)
            part = Part(**kwargs)
            db.session.add(part)
            db.session.flush()
        else:
            # ✅ НОВОЕ: безопасно обновляем имя из строки, если колонка существует
            new_name = (getattr(it, "part_name", "") or "").strip()
            if new_name and hasattr(part, "name"):
                old_name = (getattr(part, "name", "") or "").strip()
                if old_name != new_name:
                    part.name = new_name

        # увеличиваем остаток
        on_hand = _get_part_on_hand(part) + qty
        _set_part_on_hand(part, on_hand)

        # обновим last_cost / unit_cost при желании (только если такие поля есть)
        if hasattr(part, "last_cost") and it.unit_cost is not None:
            part.last_cost = float(it.unit_cost or 0)
        elif hasattr(part, "unit_cost") and it.unit_cost is not None:
            part.unit_cost = float(it.unit_cost or 0)

        # обновим location, если задана и колонка существует
        if hasattr(part, "location") and (it.location or "").strip():
            part.location = (it.location or "").strip()[:64]

    batch.status = "posted"
    batch.posted_at = datetime.utcnow()
    if current_user_id is not None:
        batch.posted_by = int(current_user_id)

    db.session.commit()
    return batch

def unpost_receiving_batch(batch_id: int, current_user_id: int | None = None):
    batch = ReceivingBatch.query.get(batch_id)
    if not batch:
        raise ValueError("Receiving batch not found")

    if batch.status != "posted":
        return batch  # уже не в posted — нечего откатывать

    # alias items -> lines уже есть; подстрахуемся на пустой список
    for it in (batch.items or []):
        pn = (it.part_number or "").strip().upper()
        qty = int(it.quantity or 0)
        if not pn or qty <= 0:
            continue

        part = Part.query.filter(func.upper(Part.part_number) == pn).first()
        if not part:
            # Детали нет — откатывать нечего по этой строке
            continue

        cur = _get_part_on_hand(part)
        new_qty = cur - qty
        if new_qty < 0:
            new_qty = 0  # защитимся от отрицательных остатков
        _set_part_on_hand(part, new_qty)

    batch.status = "draft"
    batch.posted_at = None
    # posted_by не трогаем (история может быть полезной), но можно очистить:
    # batch.posted_by = None

    db.session.commit()
    return batch

def prune_orphan_parts_by_pns(pns: set[str]) -> int:
    """
    Удаляет из каталога Part позиции по заданным PN, если:
      - остаток <= 0 (on_hand или quantity),
      - нет ссылок в приходных строках (GoodsReceiptLine), если таблица есть.
    Возвращает кол-во удалённых Part.
    """
    if not pns:
        return 0

    upper_pns = { (pn or "").strip().upper() for pn in pns if (pn or "").strip() }
    if not upper_pns:
        return 0

    removed = 0

    # Определим, доступна ли таблица строк
    try:
        from models import GoodsReceiptLine
        has_grl = True
    except Exception:
        has_grl = False

    for upn in upper_pns:
        part = Part.query.filter(func.upper(Part.part_number) == upn).first()
        if not part:
            continue

        # Остаток
        qty = 0
        if hasattr(part, "on_hand"):
            qty = int(part.on_hand or 0)
        else:
            qty = int(getattr(part, "quantity", 0) or 0)
        if qty > 0:
            continue  # есть остаток — не трогаем

        # Ссылки
        refs = 0
        if has_grl:
            try:
                refs += GoodsReceiptLine.query.filter(func.upper(GoodsReceiptLine.part_number) == upn).count()
            except Exception:
                pass
        if refs > 0:
            continue  # на деталь ещё есть ссылки — не трогаем

        db.session.delete(part)
        removed += 1

    if removed:
        db.session.commit()
    return removed

