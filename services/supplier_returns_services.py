# services/supplier_returns_services.py
from __future__ import annotations
from datetime import datetime
from typing import Dict, Any, List, Tuple

from extensions import db
from models import SupplierReturnBatch, SupplierReturnItem, Part


class SupplierReturnError(Exception):
    pass


def _find_part(pn: str) -> Part | None:
    if not pn:
        return None
    return Part.query.filter(Part.part_number == pn).first()


def recalc_batch_totals(batch: SupplierReturnBatch) -> Dict[str, Any]:
    """
    - Валидирует строки (part_number должен существовать в Part)
    - Подтягивает name/unit_cost/location из Part (если поля пусты/0)
    - Пересчитывает total_cost по строкам и агрегаты total_items/total_value
    Возвращает:
      {
        "ok": bool,
        "errors": {idx: "msg", ...},  # индекс строки в текущем порядке
      }
    """
    errors: Dict[int, str] = {}
    total_items = 0
    total_value = 0.0

    # NB: порядок как в batch.items (lazy='selectin' — стабильно по id)
    for idx, it in enumerate(batch.items or []):
        pn = (it.part_number or "").strip()
        if not pn:
            errors[idx] = "Part number required."
            it.total_cost = 0.0
            continue

        p = _find_part(pn)
        if not p:
            errors[idx] = f"Part '{pn}' not found in inventory."
            # всё равно считаем тотал по введённым данным, чтобы юзер видел цифры
            q = max(0, int(it.qty_returned or 0))
            c = float(it.unit_cost or 0.0)
            it.total_cost = round(q * c, 2)
            total_items += q
            total_value += it.total_cost
            continue

        # подставляем отсутствующие поля из инвентаря
        if not it.part_name:
            it.part_name = p.name or ""
        # если cost не задан или 0 — берём из части
        if it.unit_cost is None or float(it.unit_cost) <= 0.0:
            it.unit_cost = float(p.unit_cost or 0.0)
        # если локация пустая или "auto" — берём из части
        loc = (it.location or "").strip().lower()
        if not loc or loc == "auto":
            it.location = p.location or ""

        # нормируем количество
        q = max(0, int(it.qty_returned or 0))
        c = float(it.unit_cost or 0.0)
        it.qty_returned = q
        it.unit_cost = c
        it.total_cost = round(q * c, 2)

        total_items += q
        total_value += it.total_cost

    batch.total_items = int(total_items)
    batch.total_value = float(round(total_value, 2))

    return {"ok": len(errors) == 0, "errors": errors}


def post_batch(batch_id: int, actor: str | None = None) -> Dict[str, Any]:
    """
    Постинг возврата:
      - валидация и пересчёт;
      - уменьшение склада (Part.quantity -= qty_returned);
      - проставление статуса posted/posted_at/by.
    """
    b = SupplierReturnBatch.query.get(batch_id)
    if not b:
        raise SupplierReturnError("Batch not found.")

    if (b.status or "draft") == "posted":
        return {"ok": True, "already": True}

    # пересчёт + валидация
    info = recalc_batch_totals(b)
    if not info.get("ok"):
        # не даём постить, пока не исправят строки
        db.session.flush()
        return {"ok": False, "errors": info["errors"]}

    # проверка наличия/остатков и уменьшение
    per_row_errors: Dict[int, str] = {}
    for idx, it in enumerate(b.items or []):
        p = _find_part(it.part_number)
        if not p:
            per_row_errors[idx] = f"Part '{it.part_number}' not found."
            continue
        q = int(it.qty_returned or 0)
        if q <= 0:
            continue
        if int(p.quantity or 0) < q:
            per_row_errors[idx] = f"Not enough stock for {it.part_number} (have {p.quantity}, need {q})."
            continue
        # уменьшаем склад
        p.quantity = int(p.quantity or 0) - q
        db.session.add(p)

    if per_row_errors:
        db.session.flush()
        return {"ok": False, "errors": per_row_errors}

    # ok → ставим статус
    b.status = "posted"
    b.posted_at = datetime.utcnow()
    b.posted_by = (actor or "")[:120] if actor else None
    db.session.add(b)
    db.session.commit()

    return {"ok": True}


def unpost_batch(batch_id: int, actor: str | None = None) -> Dict[str, Any]:
    """
    Откат постинга:
      - возвращаем количество на склад (Part.quantity += qty_returned)
      - статус -> draft, чистим метаданные постинга
    """
    b = SupplierReturnBatch.query.get(batch_id)
    if not b:
        raise SupplierReturnError("Batch not found.")

    if (b.status or "draft") != "posted":
        return {"ok": False, "errors": {"_": "Only posted batch can be unposted."}}

    per_row_errors: Dict[int, str] = {}
    for idx, it in enumerate(b.items or []):
        p = _find_part(it.part_number)
        if not p:
            per_row_errors[idx] = f"Part '{it.part_number}' disappeared from inventory."
            continue
        q = int(it.qty_returned or 0)
        if q <= 0:
            continue
        p.quantity = int(p.quantity or 0) + q
        db.session.add(p)

    if per_row_errors:
        db.session.flush()
        return {"ok": False, "errors": per_row_errors}

    b.status = "draft"
    b.posted_at = None
    b.posted_by = None
    db.session.add(b)
    db.session.commit()

    return {"ok": True}
