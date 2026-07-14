# services/receiving.py
from __future__ import annotations

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
    if hasattr(part, "on_hand"):
        return int(part.on_hand or 0)
    return int(getattr(part, "quantity", 0) or 0)


def _set_part_on_hand(part, value: int):
    v = int(value)
    if v < 0:
        v = 0
    if hasattr(part, "on_hand"):
        part.on_hand = v
    else:
        part.quantity = v


# ---------- helpers: location merge / diff ----------

def _merge_locations(old_loc: str | None, new_loc: str | None) -> str:
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
    allowed = {c.name for c in Part.__table__.columns}
    data = dict(raw or {})

    if "part_name" in data and "name" in allowed and not data.get("name"):
        data["name"] = data.get("part_name")

    if "name" in allowed and not data.get("name"):
        data["name"] = data.get("part_number") or ""

    data = {k: v for k, v in data.items() if k in allowed}
    return data


# ---------- cost helpers ----------

def _safe_float(x, default=0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _ensure_line_cost_fields(it) -> tuple[float, float, float]:
    """
    Возвращает (base, extra, actual) и гарантирует, что поля заполнены.
    Правила:
      - base_unit_cost: если None -> unit_cost
      - extra_alloc_per_unit: если None -> 0.0
      - actual_unit_cost: если None -> base + extra (или unit_cost если base пустой)
    """
    unit_cost = _safe_float(getattr(it, "unit_cost", 0.0), 0.0)

    base = getattr(it, "base_unit_cost", None)
    extra = getattr(it, "extra_alloc_per_unit", None)
    actual = getattr(it, "actual_unit_cost", None)

    base_f = unit_cost if base is None else _safe_float(base, unit_cost)
    extra_f = 0.0 if extra is None else _safe_float(extra, 0.0)

    if actual is None:
        actual_f = base_f + extra_f
    else:
        actual_f = _safe_float(actual, base_f + extra_f)

    # записываем обратно (если поля реально есть)
    if hasattr(it, "base_unit_cost") and (getattr(it, "base_unit_cost", None) is None):
        it.base_unit_cost = base_f
    if hasattr(it, "extra_alloc_per_unit") and (getattr(it, "extra_alloc_per_unit", None) is None):
        it.extra_alloc_per_unit = extra_f
    if hasattr(it, "actual_unit_cost") and (getattr(it, "actual_unit_cost", None) is None):
        it.actual_unit_cost = actual_f

    return base_f, extra_f, actual_f


# ---------- core: POST (apply batch to stock & mark posted) ----------

def post_receiving_batch(
    batch_id: int,
    current_user_id: int | None = None,
):
    """
    Безопасная проводка прихода.

    Гарантии:
    - повторно posted batch не проводится;
    - пустой batch не проводится;
    - batch с некорректной строкой не проводится;
    - строки не пропускаются молча;
    - при любой ошибке выполняется rollback;
    - статус posted устанавливается только после успешной обработки всех строк;
    - applied_qty фиксирует реально применённое количество.
    """
    from flask import current_app

    try:
        batch = db.session.get(ReceivingBatch, batch_id)

        if batch is None:
            raise ValueError(
                f"Receiving batch #{batch_id} was not found."
            )

        status_now = (
            getattr(batch, "status", "") or ""
        ).strip().lower()

        # Идемпотентность: повторная проводка ничего не меняет.
        if status_now == "posted":
            current_app.logger.info(
                "[RECEIVING_POST] Batch %s is already posted. "
                "Stock update skipped.",
                batch.id,
            )
            return batch

        items = list(batch.items or [])

        # Пустой документ проводить запрещено.
        if not items:
            raise ValueError(
                f"Receiving batch #{batch.id} cannot be posted: "
                f"the batch contains no items."
            )

        validated_items = []
        validation_errors = []

        for row_number, item in enumerate(items, start=1):
            part_number = (
                getattr(item, "part_number", "") or ""
            ).strip().upper()

            raw_quantity = getattr(item, "quantity", None)

            try:
                quantity = int(raw_quantity or 0)
            except (TypeError, ValueError):
                quantity = 0

            if not part_number:
                validation_errors.append(
                    f"row {row_number}: Part # is empty"
                )
                continue

            if quantity <= 0:
                validation_errors.append(
                    f"row {row_number} ({part_number}): "
                    f"quantity must be greater than zero"
                )
                continue

            validated_items.append(
                (item, part_number, quantity)
            )

        # Не пропускаем неправильные строки молча.
        # Либо проводится весь документ, либо не проводится ничего.
        if validation_errors:
            raise ValueError(
                f"Receiving batch #{batch.id} cannot be posted. "
                + "; ".join(validation_errors)
            )

        if not validated_items:
            raise ValueError(
                f"Receiving batch #{batch.id} cannot be posted: "
                f"there are no valid items."
            )

        current_app.logger.info(
            "[RECEIVING_POST] Posting batch %s. "
            "Status before='%s', item count=%s.",
            batch.id,
            getattr(batch, "status", None),
            len(validated_items),
        )

        part_numbers = sorted({
            part_number
            for _, part_number, _ in validated_items
        })

        existing_parts = {
            (part.part_number or "").strip().upper(): part
            for part in Part.query.filter(
                func.upper(Part.part_number).in_(part_numbers)
            ).all()
        }

        for item, part_number, quantity in validated_items:
            _, _, actual_cost = _ensure_line_cost_fields(item)

            # Защита от отрицательной или некорректной себестоимости.
            actual_cost = _safe_float(actual_cost, 0.0)

            if actual_cost < 0:
                raise ValueError(
                    f"Receiving batch #{batch.id}, "
                    f"Part #{part_number}: cost cannot be negative."
                )

            part = existing_parts.get(part_number)

            if part is None:
                raw_kwargs = {
                    "part_number": part_number,
                    "part_name": (
                        getattr(item, "part_name", "") or ""
                    ).strip(),
                }

                if hasattr(Part, "supplier"):
                    raw_kwargs["supplier"] = (
                        getattr(batch, "supplier_name", "") or ""
                    ).strip()[:120]

                if hasattr(Part, "location"):
                    raw_kwargs["location"] = (
                        getattr(item, "location", "") or ""
                    ).strip().upper()[:64]

                if hasattr(Part, "unit_cost"):
                    raw_kwargs["unit_cost"] = float(actual_cost)

                kwargs = _sanitize_part_kwargs(raw_kwargs)

                part = Part(**kwargs)
                db.session.add(part)
                db.session.flush()

                existing_parts[part_number] = part
                before_quantity = 0

                current_app.logger.info(
                    "[RECEIVING_POST] Created Part '%s' "
                    "(id=%s, incoming=%s).",
                    part_number,
                    part.id,
                    quantity,
                )
            else:
                before_quantity = _get_part_on_hand(part)

                new_name = (
                    getattr(item, "part_name", "") or ""
                ).strip()

                if new_name:
                    if hasattr(part, "name"):
                        old_name = (
                            getattr(part, "name", "") or ""
                        ).strip()

                        if old_name != new_name:
                            part.name = new_name

                    elif hasattr(part, "part_name"):
                        old_name = (
                            getattr(part, "part_name", "") or ""
                        ).strip()

                        if old_name != new_name:
                            part.part_name = new_name

                current_app.logger.info(
                    "[RECEIVING_POST] Updating Part '%s' "
                    "(id=%s, before=%s, incoming=%s).",
                    part_number,
                    part.id,
                    before_quantity,
                    quantity,
                )

            after_quantity = before_quantity + quantity
            _set_part_on_hand(part, after_quantity)

            # Складская стоимость — фактическая стоимость с extra expenses.
            if hasattr(part, "last_cost"):
                part.last_cost = float(actual_cost)
            elif hasattr(part, "unit_cost"):
                part.unit_cost = float(actual_cost)

            if hasattr(part, "location"):
                incoming_location = (
                    getattr(item, "location", "") or ""
                ).strip().upper()

                if incoming_location:
                    if before_quantity > 0:
                        part.location = _merge_locations(
                            getattr(part, "location", None),
                            incoming_location,
                        )[:64]
                    else:
                        part.location = incoming_location[:64]

            if hasattr(item, "applied_qty"):
                item.applied_qty = quantity

            db.session.add(part)
            db.session.add(item)

        # Статус меняется только после успешной обработки всех строк.
        batch.status = "posted"
        batch.posted_at = datetime.utcnow()

        if (
            current_user_id is not None
            and hasattr(batch, "posted_by")
        ):
            batch.posted_by = int(current_user_id)

        db.session.add(batch)

        # Выявляем ошибки БД до окончательного commit.
        db.session.flush()
        db.session.commit()

        current_app.logger.info(
            "[RECEIVING_POST] Batch %s posted successfully. "
            "Applied items=%s, final status='%s'.",
            batch.id,
            len(validated_items),
            batch.status,
        )

        return batch

    except Exception:
        db.session.rollback()

        current_app.logger.exception(
            "[RECEIVING_POST] Batch %s posting failed. "
            "Transaction rolled back.",
            batch_id,
        )

        raise


# ---------- core: UNPOST (rollback stock using applied_qty) ----------

def unpost_receiving_batch(batch_id: int, current_user_id: int | None = None):
    """
    UNPOST (откат прихода):
    - снимаем со склада qty_applied (applied_qty fallback quantity)
    - applied_qty -> 0
    - статус batch -> draft
    """
    from flask import current_app

    batch = ReceivingBatch.query.get(batch_id)
    if not batch:
        raise ValueError(f"Receiving batch {batch_id} not found")

    curr_status = (getattr(batch, "status", "") or "").strip().lower()
    if curr_status != "posted":
        current_app.logger.info(
            "[RECEIVING_UNPOST] Batch %s not in 'posted' (status='%s'), skipping.",
            batch.id, getattr(batch, "status", None),
        )
        return batch

    current_app.logger.info(
        "[RECEIVING_UNPOST] Reverting batch %s from posted -> draft ...",
        batch.id,
    )

    for it in (batch.items or []):
        pn = (getattr(it, "part_number", "") or "").strip().upper()
        if not pn:
            continue

        qty_applied = int(getattr(it, "applied_qty", 0) or 0)
        if qty_applied <= 0:
            qty_applied = int(getattr(it, "quantity", 0) or 0)

        if qty_applied <= 0:
            continue

        part = Part.query.filter(func.upper(Part.part_number) == pn).first()
        if not part:
            continue

        before_qty = _get_part_on_hand(part)
        rollback_qty = qty_applied if qty_applied <= before_qty else before_qty
        after_qty = before_qty - rollback_qty

        _set_part_on_hand(part, after_qty)

        if hasattr(part, "location"):
            incoming_loc = (getattr(it, "location", "") or "").strip().upper()
            if after_qty == 0:
                part.location = ""
            else:
                if incoming_loc:
                    part.location = _remove_locations(getattr(part, "location", ""), incoming_loc)[:64]

        current_app.logger.info(
            "[RECEIVING_UNPOST] part %s: before=%s, rollback=%s, after=%s (qty_applied=%s)",
            pn, before_qty, rollback_qty, after_qty, qty_applied
        )

        if hasattr(it, "applied_qty"):
            it.applied_qty = 0

        db.session.add(part)
        db.session.add(it)

    batch.status = "draft"
    batch.posted_at = None

    if hasattr(batch, "unposted_by"):
        batch.unposted_by = current_user_id
    if hasattr(batch, "unposted_at"):
        batch.unposted_at = datetime.utcnow()

    db.session.add(batch)
    db.session.commit()

    current_app.logger.info(
        "[RECEIVING_UNPOST] Batch %s reverted to draft, stock rolled back safely.",
        batch.id,
    )

    return batch

