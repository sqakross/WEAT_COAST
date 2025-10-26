from datetime import datetime
from sqlalchemy import func
from extensions import db
from models import ReceivingBatch, ReceivingItem, Part

try:
    from flask import current_app
    _logger = current_app.logger
except Exception:
    import logging
    _logger = logging.getLogger(__name__)


# ---------- helpers: qty / stock field ----------

def _get_part_on_hand(part):
    """
    Возвращает текущее кол-во на складе для детали.
    Предпочитаем Part.on_hand если есть, иначе Part.quantity.
    """
    if hasattr(part, "on_hand"):
        return int(part.on_hand or 0)
    return int(getattr(part, "quantity", 0) or 0)


def _set_part_on_hand(part, value: int):
    """
    Записывает новое количество на складе (не даём уйти в минус).
    """
    v = int(value)
    if v < 0:
        v = 0
    if hasattr(part, "on_hand"):
        part.on_hand = v
    else:
        part.quantity = v


# ---------- helpers: location merge / diff ----------

def _merge_locations(old_loc: str | None, new_loc: str | None) -> str:
    """
    Склеивает локации стабильно (POST кейс):
    - upper
    - разделяем по '/'
    - без дублей
    - сохраняем порядок появления
    Примеры:
      old='C1',      new='MAR'   -> 'C1/MAR'
      old='MAR/C1',  new='C1'    -> 'MAR/C1'
      old=None,      new='AMAZ'  -> 'AMAZ'
      old='',        new=''      -> ''
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


def _remove_locations(old_loc: str | None, remove_loc: str | None) -> str:
    """
    Откат локаций (UNPOST кейс):
    - из old_loc вычитаем все токены из remove_loc
    - результат снова 'A/B/C' без дублей
    Пример:
      old_loc='C1/C3', remove_loc='C3'   -> 'C1'
      old_loc='C1/C3', remove_loc='C1'   -> 'C3'
      old_loc='C1',    remove_loc='C1'   -> '' (пусто)
    """
    if not old_loc:
        return ""
    old_tokens = []
    for token in str(old_loc).upper().split("/"):
        t = token.strip()
        if t and t not in old_tokens:
            old_tokens.append(t)

    remove_tokens = set()
    if remove_loc:
        for token in str(remove_loc).upper().split("/"):
            t = token.strip()
            if t:
                remove_tokens.add(t)

    kept = [t for t in old_tokens if t not in remove_tokens]
    return "/".join(kept)


# ---------- helpers: Part kwargs for brand new part ----------

def _sanitize_part_kwargs(raw: dict) -> dict:
    """
    Приводит kwargs к фактическим колонкам Part:
    - part_name -> name, если колонка name есть и name ещё не задано;
    - если name есть но пустой — дублируем PN;
    - удаляем ключи, которых нет у Part.
    """
    allowed = {c.name for c in Part.__table__.columns}
    data = dict(raw or {})

    # part_name -> name (если есть колонка name и name не задан)
    if "part_name" in data and "name" in allowed and not data.get("name"):
        data["name"] = data.get("part_name")

    # если name есть как колонка и не задано — подставим PN
    if "name" in allowed and not data.get("name"):
        data["name"] = data.get("part_number") or ""

    # выкинуть неизвестные ключи
    data = {k: v for k, v in data.items() if k in allowed}
    return data


# ---------- core: POST (apply batch to stock & mark posted) ----------

def post_receiving_batch(batch_id: int, current_user_id: int | None = None):
    """
    Делает приход:
    - для каждой строки batch.items плюсуем qty в Part
    - создаём Part если не было
    - мёржим location
    - апдейтим cost
    - batch.status='posted', posted_at, posted_by
    - db.session.commit()
    """
    from flask import current_app

    batch = ReceivingBatch.query.get(batch_id)
    if not batch:
        raise ValueError("Receiving batch not found")

    status_now = (batch.status or "").strip().lower()
    if status_now == "posted":
        # уже проведён — не дублируем склад
        current_app.logger.info(
            "[RECEIVING_POST] Batch %s is already posted, skipping stock update.",
            batch.id,
        )
        return batch

    current_app.logger.info(
        "[RECEIVING_POST] Posting batch %s (status before='%s') and applying to stock...",
        batch.id,
        batch.status,
    )

    # обрабатываем строки
    for it in (batch.items or []):
        pn_raw = (it.part_number or "").strip()
        qty = int(it.quantity or 0)
        if not pn_raw or qty <= 0:
            continue

        pn_upper = pn_raw.upper()
        part = Part.query.filter(func.upper(Part.part_number) == pn_upper).first()

        if not part:
            # создаём новую деталь
            raw_kwargs = {
                "part_number": pn_upper,
                "part_name": it.part_name or "",
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

            before_qty = 0
            current_app.logger.info(
                "[RECEIVING_POST] Created new Part '%s' (id=%s). Qty before=%s, incoming=%s",
                pn_upper,
                part.id,
                before_qty,
                qty,
            )
        else:
            before_qty = _get_part_on_hand(part)

            # синхроним имя (только если новое имя есть и оно отличается)
            new_name = (getattr(it, "part_name", "") or "").strip()
            if new_name and hasattr(part, "name"):
                old_name = (getattr(part, "name", "") or "").strip()
                if old_name != new_name:
                    part.name = new_name

            current_app.logger.info(
                "[RECEIVING_POST] Updating existing Part '%s' (id=%s). Qty before=%s, incoming=%s",
                pn_upper,
                part.id,
                before_qty,
                qty,
            )

        # 1) увеличить остаток
        _set_part_on_hand(part, before_qty + qty)

        # 2) обновить себестоимость
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

        # 3) слить location
        if hasattr(part, "location"):
            incoming_loc = (it.location or "").strip().upper()
            if incoming_loc:
                if before_qty > 0:
                    merged = _merge_locations(
                        getattr(part, "location", None),
                        incoming_loc,
                    )
                    part.location = merged[:64]
                else:
                    part.location = incoming_loc[:64]

    # проставляем статус posted
    batch.status = "posted"
    batch.posted_at = datetime.utcnow()
    if current_user_id is not None and hasattr(batch, "posted_by"):
        batch.posted_by = int(current_user_id)

    current_app.logger.info(
        "[RECEIVING_POST] Batch %s marked as posted. Committing.",
        batch.id,
    )

    db.session.commit()

    current_app.logger.info(
        "[RECEIVING_POST] Batch %s commit complete. Final status=%s",
        batch.id,
        batch.status,
    )

    return batch


# ---------- core: UNPOST (roll back stock & mark draft) ----------

def unpost_receiving_batch(batch_id: int, current_user_id: int | None = None):
    """
    Жёсткий откат прихода:
    - Только если батч сейчас posted.
    - Для каждой строки:
        * минусуем qty из Part (не даём в минус)
        * корректируем Part.location:
            убираем токены из этой поставки, но оставляем старые
            если остаток упал в ноль — можно вообще зачистить location
    - batch.status='draft', posted_at=None, (unposted_by/unposted_at если есть)
    - commit()
    """
    from flask import current_app

    batch = ReceivingBatch.query.get(batch_id)
    if not batch:
        raise ValueError(f"Receiving batch {batch_id} not found")

    curr_status = (batch.status or "").strip().lower()
    if curr_status != "posted":
        # Уже draft → считаем, что склад уже в нуле или ручками откатили.
        current_app.logger.info(
            "[RECEIVING_UNPOST] Batch %s not in 'posted' (status='%s'), skipping.",
            batch.id,
            batch.status,
        )
        return batch

    current_app.logger.info(
        "[RECEIVING_UNPOST] Reverting batch %s from posted -> draft ...",
        batch.id,
    )

    for it in (batch.items or []):
        pn = (it.part_number or "").strip().upper()
        qty = int(it.quantity or 0)
        if not pn or qty <= 0:
            continue

        part = Part.query.filter(func.upper(Part.part_number) == pn).first()
        if not part:
            continue  # деталь уже удалили из каталога, окей

        before_qty = _get_part_on_hand(part)
        after_qty = before_qty - qty
        if after_qty < 0:
            after_qty = 0
        _set_part_on_hand(part, after_qty)

        # location cleanup
        if hasattr(part, "location"):
            incoming_loc = (it.location or "").strip().upper()

            # если после отката остаток 0 → чистим location полностью
            if after_qty == 0:
                part.location = ""
            else:
                # иначе вырезаем только локации этой поставки
                if incoming_loc:
                    new_loc = _remove_locations(getattr(part, "location", ""), incoming_loc)
                    part.location = new_loc[:64]

    # теперь ставим батч обратно в draft
    batch.status = "draft"
    batch.posted_at = None

    # кто сделал unpost (если поля есть)
    if hasattr(batch, "unposted_by"):
        batch.unposted_by = current_user_id
    if hasattr(batch, "unposted_at"):
        batch.unposted_at = datetime.utcnow()

    db.session.commit()

    current_app.logger.info(
        "[RECEIVING_UNPOST] Batch %s reverted to draft, stock rolled back.",
        batch.id,
    )

    return batch


# ---------- cleanup helper (оставляем как у тебя) ----------

def prune_orphan_parts_by_pns(pns: set[str]) -> int:
    """
    Удаляет из каталога Part позиции по заданным PN, если:
      - остаток <= 0 (on_hand или quantity),
      - нет ссылок в приходных строках (GoodsReceiptLine), если таблица есть.
    Возвращает кол-во удалённых Part.
    """
    if not pns:
        return 0

    upper_pns = {(pn or "").strip().upper() for pn in pns if (pn or "").strip()}
    if not upper_pns:
        return 0

    removed = 0

    # есть ли таблица строк (на случай если в другой инсталляции её зовут иначе)
    try:
        from models import GoodsReceiptLine
        has_grl = True
    except Exception:
        has_grl = False

    for upn in upper_pns:
        part = Part.query.filter(func.upper(Part.part_number) == upn).first()
        if not part:
            continue

        # остаток
        if hasattr(part, "on_hand"):
            qty_val = int(part.on_hand or 0)
        else:
            qty_val = int(getattr(part, "quantity", 0) or 0)
        if qty_val > 0:
            continue  # есть остаток — не трогаем

        # ссылки
        refs = 0
        if has_grl:
            try:
                refs += GoodsReceiptLine.query.filter(
                    func.upper(GoodsReceiptLine.part_number) == upn
                ).count()
            except Exception:
                pass
        if refs > 0:
            continue  # деталь фигурирует в истории — не трогаем

        db.session.delete(part)
        removed += 1

    if removed:
        db.session.commit()
    return removed

