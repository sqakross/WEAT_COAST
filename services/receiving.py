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

def _merge_locations(old_loc: str | None, new_loc: str | None) -> str:
    """
    Склеивает локации стабильно:
    - переводит в UPPER
    - разбивает по '/'
    - убирает пустые и дубликаты
    - сохраняет порядок появления
    Примеры:
      old='C1'   , new='MAR'      -> 'C1/MAR'
      old='MAR/C1', new='C1'      -> 'MAR/C1'
      old=None   , new='AMAZ'     -> 'AMAZ'
      old=''     , new=''         -> ''
    """
    out = []
    for raw in (old_loc, new_loc):
        if not raw:
            continue
        for token in str(raw).upper().split("/"):
            t = token.strip()
            if t and t not in out:
                out.append(t)
    return "/".join(out)


def post_receiving_batch(batch_id: int, current_user_id: int | None = None):
    batch = ReceivingBatch.query.get(batch_id)
    if not batch:
        raise ValueError("Receiving batch not found")

    # не делаем двойной приход
    if (batch.status or "").lower() == "posted":
        return batch

    # идём по позициям батча
    for it in (batch.items or []):  # у тебя alias items -> lines
        pn  = (it.part_number or "").strip()
        qty = int(it.quantity or 0)
        if not pn or qty <= 0:
            continue

        pn_upper = pn.upper()

        # найти Part по PN (case-insensitive)
        part = Part.query.filter(func.upper(Part.part_number) == pn_upper).first()

        if not part:
            # --- создаём новый Part, опираясь на твою _sanitize_part_kwargs
            raw_kwargs = {
                "part_number": pn_upper,
                "part_name": it.part_name or "",  # _sanitize_part_kwargs скопирует это в name, если надо
            }
            if hasattr(Part, "supplier"):
                raw_kwargs["supplier"] = (batch.supplier_name or "")[:120]
            if hasattr(Part, "location"):
                raw_kwargs["location"] = (it.location or "").strip()[:64]
            if hasattr(Part, "unit_cost") and it.unit_cost is not None:
                raw_kwargs["unit_cost"] = float(it.unit_cost or 0)

            kwargs = _sanitize_part_kwargs(raw_kwargs)
            part = Part(**kwargs)
            db.session.add(part)
            db.session.flush()

            before_qty = 0  # у нового товара склад до прихода = 0
        else:
            # уже был в каталоге
            before_qty = _get_part_on_hand(part)

            # безопасный апдейт имени детали
            new_name = (getattr(it, "part_name", "") or "").strip()
            if new_name and hasattr(part, "name"):
                old_name = (getattr(part, "name", "") or "").strip()
                if old_name != new_name:
                    part.name = new_name

        # === теперь инвентарь ===

        # 1. увеличиваем остаток
        _set_part_on_hand(part, before_qty + qty)

        # 2. обновляем cost (last_cost / unit_cost), если пришёл
        if it.unit_cost is not None:
            try:
                cost_val = float(it.unit_cost or 0)
            except Exception:
                cost_val = None
            if cost_val is not None:
                if hasattr(part, "last_cost"):
                    part.last_cost = cost_val
                elif hasattr(part, "unit_cost"):
                    part.unit_cost = cost_val

        # 3. обновляем location, НО НЕ ПЕРЕЗАТИРАЕМ
        if hasattr(part, "location"):
            incoming_loc = (it.location or "").strip().upper()
            if incoming_loc:
                if before_qty > 0:
                    # товар уже был на складе -> мержим старое+новое
                    merged = _merge_locations(getattr(part, "location", None), incoming_loc)
                    part.location = merged[:64]  # safety truncate
                else:
                    # товара не было (before_qty == 0) -> просто ставим эту локацию
                    part.location = incoming_loc[:64]

    # проставить статус батчу
    batch.status = "posted"
    batch.posted_at = datetime.utcnow()
    if current_user_id is not None:
        batch.posted_by = int(current_user_id)

    db.session.commit()
    return batch

from datetime import datetime
from sqlalchemy import func
from extensions import db
from models import ReceivingBatch, ReceivingItem, Part

def unpost_receiving_batch(batch_id: int, current_user_id: int | None = None):
    """
    Жёсткий откат прихода:
    - ВСЕГДА вычитаем qty из Part.quantity по всем строкам батча
    - переводим батч в draft
    - не даём уйти в минус
    - не делаем никаких ранних return
    """

    batch = ReceivingBatch.query.get(batch_id)
    if not batch:
        raise ValueError(f"Receiving batch {batch_id} not found")

    # пройти по всем строкам партии и снять количество
    for it in (batch.items or []):
        pn  = (it.part_number or "").strip().upper()
        qty = int(it.quantity or 0)
        if not pn or qty <= 0:
            continue

        # ищем Part по PN
        part = Part.query.filter(func.upper(Part.part_number) == pn).first()
        if not part:
            # детали нет в каталоге — нечего откатывать
            continue

        before_qty = int(part.quantity or 0)
        after_qty  = before_qty - qty
        if after_qty < 0:
            after_qty = 0

        part.quantity = after_qty
        # цену / location не трогаем специально

    # теперь ставим батч обратно в draft
    batch.status = "draft"
    batch.posted_at = None

    # логируем кто отменил (если такие поля есть)
    if hasattr(batch, "unposted_by"):
        batch.unposted_by = current_user_id
    if hasattr(batch, "unposted_at"):
        batch.unposted_at = datetime.utcnow()

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

