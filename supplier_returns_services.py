"""
Supplier Returns: minimal services (apply / rollback) без правок существующего кода.
- Списывает Part.quantity при POST (возврат поставщику)
- Возвращает Part.quantity при UNPOST
- Проверяет остатки; матчит по part_number и (если указана) location
"""

from datetime import datetime
from typing import Optional

from extensions import db
from models import SupplierReturnBatch, SupplierReturnItem, Part


class SupplierReturnError(Exception):
    """Бизнес-ошибка возврата поставщику."""


def _find_part(item: SupplierReturnItem) -> Optional[Part]:
    """
    Ищем Part по part_number (+ optional location).
    Если у позиции указана location — требуем точное совпадение.
    Если location пустая — ищем только по part_number (берём первый).
    """
    q = Part.query.filter(Part.part_number == item.part_number)
    loc = (item.location or "").strip()
    if loc:
        q = q.filter(Part.location == loc)
    return q.first()


def _require_stock_for_item(item: SupplierReturnItem):
    """Проверяем, что на складе достаточно штучного количества для списания."""
    if (item.qty_returned or 0) <= 0:
        raise SupplierReturnError(f"Qty must be > 0 for part {item.part_number}")

    part = _find_part(item)
    if not part:
        loc = (item.location or "").strip()
        if loc:
            raise SupplierReturnError(f"Part not found: {item.part_number} at location {loc}")
        raise SupplierReturnError(f"Part not found: {item.part_number}")

    qty = int(part.quantity or 0)
    need = int(item.qty_returned or 0)
    if qty < need:
        raise SupplierReturnError(
            f"Insufficient stock for {item.part_number} "
            f"(have {qty}, need {need}{' at '+part.location if part.location else ''})"
        )


def apply_supplier_return(batch_id: int, user_name: Optional[str] = None) -> SupplierReturnBatch:
    """
    POST возврата поставщику:
    - проверяем, что хватает остатков по каждой позиции
    - уменьшаем Part.quantity на qty_returned
    - помечаем batch.status='posted', проставляем posted_at/by

    Возвращает обновлённый SupplierReturnBatch.
    """
    b = SupplierReturnBatch.query.get(batch_id)
    if not b:
        raise SupplierReturnError(f"Batch not found: {batch_id}")

    if (b.status or "").lower() == "posted":
        return b  # уже применён

    # 1) валидация остатков по всем строкам
    for it in (b.items or []):
        _require_stock_for_item(it)

    # 2) транзакционное списание
    with db.session.begin():
        # уменьшаем остатки
        for it in (b.items or []):
            part = _find_part(it)
            # `_require_stock_for_item` гарантирует существование и достаточность
            part.quantity = int(part.quantity or 0) - int(it.qty_returned or 0)

        # статус и метаданные
        b.status = "posted"
        b.posted_at = datetime.utcnow()
        b.posted_by = (user_name or "").strip()[:120] if user_name else b.posted_by

        # обновим агрегаты на всякий случай (необязательно, но полезно)
        total_items = len(b.items or [])
        total_value = 0.0
        for it in (b.items or []):
            q = int(it.qty_returned or 0)
            c = float(it.unit_cost or 0.0)
            it.total_cost = float(q * c)
            total_value += it.total_cost
        b.total_items = total_items
        b.total_value = float(total_value)

    return b


def rollback_supplier_return(batch_id: int, user_name: Optional[str] = None) -> SupplierReturnBatch:
    """
    UNPOST возврата поставщику:
    - увеличиваем Part.quantity обратно на qty_returned
    - возвращаем статус batch.status='draft'
    """
    b = SupplierReturnBatch.query.get(batch_id)
    if not b:
        raise SupplierReturnError(f"Batch not found: {batch_id}")

    if (b.status or "").lower() != "posted":
        return b  # откатывать нечего

    with db.session.begin():
        for it in (b.items or []):
            part = _find_part(it)
            # Если вдруг Part уже удалён — пропускаем (мягкий откат)
            if part:
                part.quantity = int(part.quantity or 0) + int(it.qty_returned or 0)

        b.status = "draft"
        b.posted_at = None
        # posted_by не трогаем, чтобы оставался след кто постил; при желании можно чистить
        # b.posted_by = None

    return b
