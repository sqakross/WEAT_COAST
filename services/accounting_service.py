from __future__ import annotations
from datetime import date
from sqlalchemy import func
from extensions import db
from models import IssuedPartRecord, IssuedBatch, Part
from dataclasses import dataclass


@dataclass(frozen=True)
class TechnicianSummaryRow:
    technician: str
    supplier: str
    base_total: float
    extra_total: float
    actual_total: float
    returns_total: float
    net_total: float
    qty_total: int


def get_technician_summary(
    date_from: date | None = None,
    date_to: date | None = None,
    supplier: str | None = None,
) -> list[TechnicianSummaryRow]:
    """
    Read-only accounting summary.
    Safe: does not update DB.
    Fast: SQL aggregation, no Python loops over records.
    """

    technician_expr = func.coalesce(IssuedBatch.issued_to, IssuedPartRecord.issued_to, "UNKNOWN")

    # Пока supplier точный не всегда доступен, потому что source_receipt_line_id старых records = None.
    # Поэтому временно ставим UNKNOWN. Потом добавим join к GoodsReceipt/GoodsReceiptLine.
    supplier_expr = func.coalesce(IssuedPartRecord.cost_source, "UNKNOWN")

    qty_expr = func.coalesce(func.sum(IssuedPartRecord.quantity), 0)

    base_total_expr = func.coalesce(
        func.sum(IssuedPartRecord.quantity * IssuedPartRecord.unit_cost_at_issue),
        0.0,
    )

    q = (
        db.session.query(
            technician_expr.label("technician"),
            supplier_expr.label("supplier"),
            qty_expr.label("qty_total"),
            base_total_expr.label("base_total"),
        )
        .outerjoin(IssuedBatch, IssuedBatch.id == IssuedPartRecord.batch_id)
        .outerjoin(Part, Part.id == IssuedPartRecord.part_id)
    )

    if date_from:
        q = q.filter(func.date(IssuedPartRecord.issue_date) >= date_from)

    if date_to:
        q = q.filter(func.date(IssuedPartRecord.issue_date) <= date_to)

    if supplier:
        q = q.filter(supplier_expr == supplier)

    q = (
        q.group_by(technician_expr, supplier_expr)
        .order_by(technician_expr.asc())
    )

    rows: list[TechnicianSummaryRow] = []

    for r in q.all():
        base_total = float(r.base_total or 0.0)

        rows.append(
            TechnicianSummaryRow(
                technician=r.technician or "UNKNOWN",
                supplier=r.supplier or "UNKNOWN",
                base_total=base_total,
                extra_total=0.0,
                actual_total=base_total,
                returns_total=0.0,
                net_total=base_total,
                qty_total=int(r.qty_total or 0),
            )
        )

    return rows

@dataclass(frozen=True)
class AccountingDataQuality:
    total_issued: int
    linked_to_receipt: int
    not_linked_to_receipt: int
    linked_percent: float
    by_cost_source: list[tuple[str, int]]


def get_accounting_data_quality() -> AccountingDataQuality:
    total_issued = db.session.query(func.count(IssuedPartRecord.id)).scalar() or 0

    linked_to_receipt = (
        db.session.query(func.count(IssuedPartRecord.id))
        .filter(IssuedPartRecord.source_receipt_line_id.isnot(None))
        .scalar()
        or 0
    )

    not_linked = max(0, total_issued - linked_to_receipt)

    linked_percent = (
        round((linked_to_receipt / total_issued) * 100, 2)
        if total_issued
        else 0.0
    )

    source_rows = (
        db.session.query(
            func.coalesce(IssuedPartRecord.cost_source, "UNKNOWN").label("source"),
            func.count(IssuedPartRecord.id).label("count"),
        )
        .group_by(func.coalesce(IssuedPartRecord.cost_source, "UNKNOWN"))
        .order_by(func.count(IssuedPartRecord.id).desc())
        .all()
    )

    by_cost_source = [
        (str(r.source or "UNKNOWN"), int(r.count or 0))
        for r in source_rows
    ]

    return AccountingDataQuality(
        total_issued=int(total_issued),
        linked_to_receipt=int(linked_to_receipt),
        not_linked_to_receipt=int(not_linked),
        linked_percent=float(linked_percent),
        by_cost_source=by_cost_source,
    )

@dataclass(frozen=True)
class InvoiceDetailRow:
    issue_date: str
    technician: str
    job_number: str
    supplier: str
    supplier_invoice: str
    part_number: str
    part_name: str
    qty: int
    base_unit_cost: float
    extra_per_unit: float
    actual_unit_cost: float
    line_base_total: float
    line_extra_total: float
    line_actual_total: float
    cost_source: str
    link_status: str


def get_invoice_detail_rows(
    date_from: date | None = None,
    date_to: date | None = None,
    technician: str | None = None,
    supplier: str | None = None,
    limit: int = 500,
) -> list[InvoiceDetailRow]:
    """
    Read-only accounting detail.
    Does not update DB.
    Linked rows use GoodsReceipt/GoodsReceiptLine.
    Unlinked old rows stay visible as UNLINKED.
    """

    from models import GoodsReceipt, GoodsReceiptLine

    q = (
        db.session.query(
            IssuedPartRecord.issue_date.label("issue_date"),
            func.coalesce(IssuedBatch.issued_to, IssuedPartRecord.issued_to, "UNKNOWN").label("technician"),
            func.coalesce(IssuedBatch.reference_job, IssuedPartRecord.reference_job, "").label("job_number"),

            func.coalesce(GoodsReceipt.supplier_name, "UNKNOWN").label("supplier"),
            func.coalesce(GoodsReceipt.invoice_number, IssuedPartRecord.inv_ref, "").label("supplier_invoice"),

            Part.part_number.label("part_number"),
            Part.name.label("part_name"),

            IssuedPartRecord.quantity.label("qty"),
            IssuedPartRecord.unit_cost_at_issue.label("issued_unit_cost"),
            IssuedPartRecord.cost_source.label("cost_source"),

            GoodsReceiptLine.base_unit_cost.label("base_unit_cost"),
            GoodsReceiptLine.extra_alloc_per_unit.label("extra_per_unit"),
            GoodsReceiptLine.actual_unit_cost.label("actual_unit_cost"),
            GoodsReceiptLine.unit_cost.label("receipt_unit_cost"),
            IssuedPartRecord.source_receipt_line_id.label("source_receipt_line_id"),
        )
        .join(Part, Part.id == IssuedPartRecord.part_id)
        .outerjoin(IssuedBatch, IssuedBatch.id == IssuedPartRecord.batch_id)
        .outerjoin(
            GoodsReceiptLine,
            GoodsReceiptLine.id == IssuedPartRecord.source_receipt_line_id,
        )
        .outerjoin(
            GoodsReceipt,
            GoodsReceipt.id == GoodsReceiptLine.goods_receipt_id,
        )
    )

    if date_from:
        q = q.filter(func.date(IssuedPartRecord.issue_date) >= date_from)

    if date_to:
        q = q.filter(func.date(IssuedPartRecord.issue_date) <= date_to)

    if technician:
        like_value = f"%{technician.strip()}%"
        q = q.filter(
            func.coalesce(IssuedBatch.issued_to, IssuedPartRecord.issued_to, "")
            .ilike(like_value)
        )

    if supplier:
        q = q.filter(GoodsReceipt.supplier_name.ilike(f"%{supplier.strip()}%"))

    q = q.order_by(IssuedPartRecord.issue_date.desc()).limit(limit)

    rows: list[InvoiceDetailRow] = []

    for r in q.all():
        qty = int(r.qty or 0)
        cost_source = (r.cost_source or "UNKNOWN").strip()

        is_linked = r.source_receipt_line_id is not None

        if cost_source == "BASE_RETURN":
            link_status = "RETURN"
        elif is_linked:
            link_status = "LINKED"
        elif cost_source in {"latest", "fallback_part_unit_cost", "fifo", "receipt_latest_fallback"}:
            link_status = "FALLBACK"
        else:
            link_status = "UNLINKED"

        base_unit = float(
            r.base_unit_cost
            if r.base_unit_cost is not None
            else r.issued_unit_cost or 0.0
        )

        extra_unit = float(r.extra_per_unit or 0.0)

        actual_unit = float(
            r.actual_unit_cost
            if r.actual_unit_cost is not None
            else r.receipt_unit_cost
            if r.receipt_unit_cost is not None
            else r.issued_unit_cost or 0.0
        )

        rows.append(
            InvoiceDetailRow(
                issue_date=r.issue_date.strftime("%Y-%m-%d") if r.issue_date else "",
                technician=r.technician or "UNKNOWN",
                job_number=r.job_number or "",
                supplier=r.supplier or "UNKNOWN",
                supplier_invoice=str(r.supplier_invoice or ""),
                part_number=r.part_number or "",
                part_name=r.part_name or "",
                qty=qty,
                base_unit_cost=round(base_unit, 4),
                extra_per_unit=round(extra_unit, 4),
                actual_unit_cost=round(actual_unit, 4),
                line_base_total=round(qty * base_unit, 2),
                line_extra_total=round(qty * extra_unit, 2),
                line_actual_total=round(qty * actual_unit, 2),
                cost_source=cost_source,
                link_status=link_status,
            )
        )

    return rows

from models import (
    TechnicianLedgerEntry,
    TechnicianPayment,
    TechnicianPaymentAllocation,
)


def _sync_ledger_status(entry: TechnicianLedgerEntry) -> None:
    amount = round(float(entry.amount or 0.0), 2)
    paid = round(float(entry.paid_amount or 0.0), 2)
    remaining = round(float(entry.remaining_amount or 0.0), 2)

    entry.amount = amount
    entry.paid_amount = paid
    entry.remaining_amount = max(0.0, remaining)

    if entry.remaining_amount <= 0:
        entry.status = "paid"
        entry.remaining_amount = 0.0
    elif entry.paid_amount > 0:
        entry.status = "partial"
    else:
        entry.status = "open"


def create_technician_payment_fifo(
    technician_name: str,
    amount: float,
    payment_date: date,
    method: str | None = None,
    reference: str | None = None,
    note: str | None = None,
    created_by: int | None = None,
) -> TechnicianPayment:
    """
    Create technician payment and allocate it FIFO to oldest open ledger debt.
    Safe: affects only technician ledger/payment tables.
    """

    tech = (technician_name or "").strip()
    pay_amount = round(float(amount or 0.0), 2)

    if not tech:
        raise ValueError("Technician name is required")

    if pay_amount <= 0:
        raise ValueError("Payment amount must be greater than zero")

    try:
        from datetime import datetime

        payment = TechnicianPayment(
            technician_name=tech,
            payment_date=payment_date,
            amount=pay_amount,
            unapplied_amount=pay_amount,
            method=(method or "").strip() or None,
            reference=(reference or "").strip() or None,
            note=(note or "").strip() or None,
            created_by=created_by,
            status="posted",
            posted_at=datetime.utcnow(),
            posted_by=created_by,
            voided=False,
        )

        db.session.add(payment)
        db.session.flush()

        remaining_payment = pay_amount

        open_entries = (
            db.session.query(TechnicianLedgerEntry)
            .filter(
                TechnicianLedgerEntry.technician_name == tech,
                TechnicianLedgerEntry.status.in_(["open", "partial"]),
                TechnicianLedgerEntry.remaining_amount > 0,
            )
            .order_by(
                TechnicianLedgerEntry.entry_date.asc(),
                TechnicianLedgerEntry.id.asc(),
            )
            .all()
        )

        for entry in open_entries:
            if remaining_payment <= 0:
                break

            entry_remaining = round(float(entry.remaining_amount or 0.0), 2)

            if entry_remaining <= 0:
                continue

            applied = round(min(remaining_payment, entry_remaining), 2)

            allocation = TechnicianPaymentAllocation(
                payment_id=payment.id,
                ledger_entry_id=entry.id,
                amount=applied,
            )

            db.session.add(allocation)

            entry.paid_amount = round(float(entry.paid_amount or 0.0) + applied, 2)
            entry.remaining_amount = round(entry_remaining - applied, 2)

            _sync_ledger_status(entry)

            remaining_payment = round(remaining_payment - applied, 2)

        payment.unapplied_amount = round(remaining_payment, 2)

        db.session.commit()
        return payment

    except Exception:
        db.session.rollback()
        raise

def void_technician_payment(
    payment_id: int,
    void_reason: str,
    voided_by: int | None = None,
) -> TechnicianPayment:
    """
    Void a posted technician payment.

    The payment and its allocation history are preserved.
    Applied amounts are reversed from ledger entries.
    """

    from datetime import datetime

    reason = (void_reason or "").strip()

    if not reason:
        raise ValueError("Void reason is required")

    payment = (
        TechnicianPayment.query
        .filter(TechnicianPayment.id == int(payment_id))
        .first()
    )

    if payment is None:
        raise ValueError("Payment was not found")

    if payment.voided or (payment.status or "").lower() == "void":
        raise ValueError(
            f"Payment #{payment.id} has already been voided"
        )

    if (payment.status or "").lower() != "posted":
        raise ValueError(
            "Only a posted payment can be voided"
        )

    try:
        for allocation in payment.allocations or []:
            entry = allocation.ledger_entry

            if entry is None:
                raise ValueError(
                    f"Ledger entry for allocation "
                    f"#{allocation.id} was not found"
                )

            applied = round(
                float(allocation.amount or 0.0),
                2,
            )

            if applied <= 0:
                continue

            current_paid = round(
                float(entry.paid_amount or 0.0),
                2,
            )

            if applied > current_paid:
                raise ValueError(
                    f"Cannot void payment #{payment.id}: "
                    f"allocation ${applied:.2f} exceeds the "
                    f"current paid amount for ledger entry "
                    f"#{entry.id}."
                )

            original_amount = round(
                float(entry.amount or 0.0),
                2,
            )

            new_paid = round(
                current_paid - applied,
                2,
            )

            new_remaining = round(
                original_amount - new_paid,
                2,
            )

            entry.paid_amount = max(0.0, new_paid)
            entry.remaining_amount = max(
                0.0,
                new_remaining,
            )

            _sync_ledger_status(entry)

        payment.status = "void"
        payment.voided = True
        payment.voided_at = datetime.utcnow()
        payment.voided_by = voided_by
        payment.void_reason = reason[:255]

        # После void деньги больше не считаются действующим payment.
        # Allocations остаются в БД как исторический audit trail.
        payment.unapplied_amount = round(
            float(payment.amount or 0.0),
            2,
        )

        db.session.commit()

        return payment

    except Exception:
        db.session.rollback()
        raise

def create_ledger_charge_from_issued_record(
    record,
    invoice_number: int | str | None = None,
    work_order_id: int | None = None,
    default_technician: str | None = None,
    default_job: str | None = None,
    note_prefix: str | None = None,
):
    """
    Create or refresh CHARGE ledger entry from IssuedPartRecord.
    Does NOT commit.
    """

    from datetime import date
    from extensions import db
    from models import TechnicianLedgerEntry

    if not record:
        return None

    qty = int(getattr(record, "quantity", 0) or 0)
    unit_cost = float(getattr(record, "unit_cost_at_issue", 0.0) or 0.0)
    amount = round(qty * unit_cost, 2)

    if qty <= 0 or amount <= 0:
        return None

    issue_date = getattr(record, "issue_date", None)

    technician = (
        getattr(record, "issued_to", None)
        or default_technician
        or "UNKNOWN"
    )

    job_number = (
        getattr(record, "reference_job", None)
        or default_job
        or None
    )

    supplier_invoice = (
        getattr(record, "inv_ref", None)
        or str(invoice_number or "")
        or None
    )

    note = note_prefix or "Issued part"
    if invoice_number:
        note = f"{note} invoice #{invoice_number}"
    if work_order_id:
        note = f"{note} / WO #{work_order_id}"

    existing = TechnicianLedgerEntry.query.filter_by(
        issued_part_record_id=record.id,
        entry_type="CHARGE",
    ).first()

    if existing:
        existing.supplier_invoice = (
            str(supplier_invoice).strip()
            if supplier_invoice
            else existing.supplier_invoice
        )
        existing.job_number = (
            str(job_number).strip()
            if job_number
            else existing.job_number
        )
        existing.note = note[:500]

        # Safe refresh only if no payment was applied yet
        if float(existing.paid_amount or 0.0) <= 0:
            existing.amount = amount
            existing.remaining_amount = amount
            existing.status = "open" if amount > 0 else "paid"

        return existing

    entry = TechnicianLedgerEntry(
        technician_name=str(technician).strip(),
        entry_date=issue_date.date() if issue_date else date.today(),
        entry_type="CHARGE",
        status="open",

        amount=amount,
        paid_amount=0.0,
        remaining_amount=amount,

        supplier_name=None,
        supplier_invoice=str(supplier_invoice).strip() if supplier_invoice else None,
        job_number=str(job_number).strip() if job_number else None,

        issued_part_record_id=record.id,
        reference_type="issued_part_record",
        reference_id=record.id,

        note=note[:500],
    )

    db.session.add(entry)
    return entry


def create_ledger_charges_for_records(
    records: list,
    invoice_number: int | str | None = None,
    work_order_id: int | None = None,
    default_technician: str | None = None,
    default_job: str | None = None,
    note_prefix: str | None = None,
) -> list:
    """
    Bulk helper. Does NOT commit.
    """

    entries = []

    for record in records or []:
        entry = create_ledger_charge_from_issued_record(
            record=record,
            invoice_number=invoice_number,
            work_order_id=work_order_id,
            default_technician=default_technician,
            default_job=default_job,
            note_prefix=note_prefix,
        )
        if entry:
            entries.append(entry)

    return entries

@dataclass(frozen=True)
class TechnicianBalanceRow:
    technician: str
    total_amount: float
    paid_amount: float
    remaining_amount: float
    open_count: int


def get_technician_balances() -> list[TechnicianBalanceRow]:
    from models import TechnicianLedgerEntry

    rows = (
        db.session.query(
            TechnicianLedgerEntry.technician_name.label("technician"),
            func.coalesce(func.sum(TechnicianLedgerEntry.amount), 0.0).label("total_amount"),
            func.coalesce(func.sum(TechnicianLedgerEntry.paid_amount), 0.0).label("paid_amount"),
            func.coalesce(func.sum(TechnicianLedgerEntry.remaining_amount), 0.0).label("remaining_amount"),
            func.count(
                func.distinct(TechnicianLedgerEntry.supplier_invoice)
            ).label("open_count"),
        )
        .filter(TechnicianLedgerEntry.status.in_(["open", "partial"]))
        .group_by(TechnicianLedgerEntry.technician_name)
        .order_by(func.sum(TechnicianLedgerEntry.remaining_amount).desc())
        .all()
    )

    return [
        TechnicianBalanceRow(
            technician=r.technician or "UNKNOWN",
            total_amount=round(float(r.total_amount or 0.0), 2),
            paid_amount=round(float(r.paid_amount or 0.0), 2),
            remaining_amount=round(float(r.remaining_amount or 0.0), 2),
            open_count=int(r.open_count or 0),
        )
        for r in rows
    ]

def create_technician_adjustment(
    technician_name: str,
    amount: float,
    adjustment_type: str,
    reason: str | None = None,
    created_by: int | None = None,
):
    """
    Create manual technician accounting adjustment.
    Does NOT touch old ledger entries.
    """

    from datetime import date
    from extensions import db
    from models import TechnicianLedgerEntry

    tech = (technician_name or "").strip()
    adj_type = (adjustment_type or "").strip().upper()
    amt = round(float(amount or 0.0), 2)

    allowed = {
        "OPENING_BALANCE",
        "ADJUSTMENT_PLUS",
        "ADJUSTMENT_MINUS",
    }

    if not tech:
        raise ValueError("Technician name is required")

    if adj_type not in allowed:
        raise ValueError("Invalid adjustment type")

    if amt <= 0:
        raise ValueError("Amount must be greater than zero")

    if adj_type == "ADJUSTMENT_MINUS":
        amt = -abs(amt)
    else:
        amt = abs(amt)

    entry = TechnicianLedgerEntry(
        technician_name=tech,
        entry_date=date.today(),
        entry_type=adj_type,
        status="open",
        amount=amt,
        paid_amount=0.0,
        remaining_amount=amt,
        supplier_name=None,
        supplier_invoice=None,
        job_number=None,
        issued_part_record_id=None,
        reference_type="manual_adjustment",
        reference_id=None,
        note=(reason or adj_type)[:500],
        adjustment_reason=(reason or "")[:255],
        created_by=created_by,
    )

    db.session.add(entry)
    db.session.commit()

    return entry

def create_technician_opening_balance(
    technician_name: str,
    amount: float,
    opening_date: date,
    note: str | None = None,
    created_by: int | None = None,
) -> TechnicianLedgerEntry:
    """
    Create one opening balance for a technician.

    Opening balance is stored as an immutable ledger entry.
    Existing accounting history is not rewritten.
    """

    tech = (technician_name or "").strip()
    opening_amount = round(float(amount or 0.0), 2)
    clean_note = (note or "").strip()

    if not tech:
        raise ValueError("Technician name is required")

    if opening_amount <= 0:
        raise ValueError("Opening balance must be greater than zero")

    if not opening_date:
        raise ValueError("Opening balance date is required")

    existing = (
        TechnicianLedgerEntry.query
        .filter(
            TechnicianLedgerEntry.technician_name == tech,
            TechnicianLedgerEntry.entry_type == "OPENING_BALANCE",
            TechnicianLedgerEntry.voided == False,
        )
        .first()
    )

    if existing:
        raise ValueError(
            f"Opening balance already exists for {tech}. "
            f"Entry #{existing.id}. Use an adjustment instead."
        )

    try:
        entry = TechnicianLedgerEntry(
            technician_name=tech,
            entry_date=opening_date,
            entry_type="OPENING_BALANCE",
            status="open",

            amount=opening_amount,
            paid_amount=0.0,
            remaining_amount=opening_amount,

            supplier_name=None,
            supplier_invoice=None,
            job_number=None,

            adjustment_reason=None,

            issued_part_record_id=None,
            reference_type="opening_balance",
            reference_id=None,

            note=clean_note[:500] or "Opening balance",
            created_by=created_by,
            voided=False,
        )

        db.session.add(entry)
        db.session.commit()

        return entry

    except Exception:
        db.session.rollback()
        raise

@dataclass(frozen=True)
class TechnicianPaymentAllocationRow:
    ledger_entry_id: int
    entry_date: str
    supplier_invoice: str
    job_number: str
    amount: float


@dataclass(frozen=True)
class TechnicianPaymentRow:
    id: int
    document_number: str

    payment_date: str
    amount: float
    applied_amount: float
    historical_applied_amount: float
    unapplied_amount: float

    method: str
    reference: str
    note: str
    status: str

    created_by_name: str
    created_at: str

    posted_by_name: str
    posted_at: str

    voided_by_name: str
    voided_at: str
    void_reason: str

    allocations: list[TechnicianPaymentAllocationRow]

@dataclass(frozen=True)
class TechnicianLedgerRow:
    id: int
    entry_date: str
    entry_type: str

    supplier_name: str | None
    supplier_invoice: str | None
    job_number: str | None

    amount: float
    paid_amount: float
    remaining_amount: float

    status: str
    note: str | None

    created_by_name: str
    created_at: str

@dataclass(frozen=True)
class TechnicianSummary:
    technician: str

    charges: float

    payments: float

    adjustments: float

    opening_balance: float

    remaining: float

    open_invoices: int

def get_technician_payments(
    technician_name: str,
) -> list[TechnicianPaymentRow]:
    """
    Return technician payment history with FIFO allocations
    and complete audit information.

    Read-only: does not modify accounting records.
    """

    from models import User, utc_to_local

    tech = (technician_name or "").strip()

    if not tech:
        return []

    payments = (
        TechnicianPayment.query
        .filter(
            TechnicianPayment.technician_name == tech,
        )
        .order_by(
            TechnicianPayment.payment_date.desc(),
            TechnicianPayment.created_at.desc(),
            TechnicianPayment.id.desc(),
        )
        .all()
    )

    # Собираем user IDs одним запросом, чтобы не создавать N+1 queries.
    user_ids = set()

    for payment in payments:
        for user_id in (
            payment.created_by,
            payment.posted_by,
            payment.voided_by,
        ):
            if user_id:
                user_ids.add(int(user_id))

    users_by_id = {}

    if user_ids:
        users = (
            User.query
            .filter(User.id.in_(user_ids))
            .all()
        )

        users_by_id = {
            int(user.id): user.username
            for user in users
        }

    def username_for(user_id) -> str:
        if not user_id:
            return ""

        return users_by_id.get(
            int(user_id),
            f"User #{user_id}",
        )

    def format_datetime(value) -> str:
        if not value:
            return ""

        local_value = utc_to_local(value)

        if not local_value:
            return ""

        return local_value.strftime("%m/%d/%Y %I:%M %p")

    rows: list[TechnicianPaymentRow] = []

    for payment in payments:
        allocation_rows = []

        historical_applied = 0.0

        for allocation in payment.allocations or []:
            ledger_entry = allocation.ledger_entry

            allocation_amount = round(
                float(allocation.amount or 0.0),
                2,
            )

            historical_applied = round(
                historical_applied + allocation_amount,
                2,
            )

            allocation_rows.append(
                TechnicianPaymentAllocationRow(
                    ledger_entry_id=(
                        ledger_entry.id
                        if ledger_entry
                        else allocation.ledger_entry_id
                    ),
                    entry_date=(
                        ledger_entry.entry_date.strftime("%m/%d/%Y")
                        if ledger_entry and ledger_entry.entry_date
                        else ""
                    ),
                    supplier_invoice=(
                        ledger_entry.supplier_invoice
                        if ledger_entry
                        else ""
                    ) or "",
                    job_number=(
                        ledger_entry.job_number
                        if ledger_entry
                        else ""
                    ) or "",
                    amount=allocation_amount,
                )
            )

        allocation_rows.sort(
            key=lambda row: (
                row.entry_date,
                row.ledger_entry_id,
            )
        )

        amount = round(
            float(payment.amount or 0.0),
            2,
        )

        unapplied = round(
            float(payment.unapplied_amount or 0.0),
            2,
        )

        is_void = bool(
            payment.voided
            or (payment.status or "").strip().lower() == "void"
        )

        if is_void:
            active_applied = 0.0
            status = "void"
        else:
            active_applied = round(
                amount - unapplied,
                2,
            )
            status = (payment.status or "").strip().lower()

        rows.append(
            TechnicianPaymentRow(
                id=payment.id,
                document_number=f"PAY-{payment.id:06d}",

                payment_date=(
                    payment.payment_date.strftime("%m/%d/%Y")
                    if payment.payment_date
                    else ""
                ),

                amount=amount,
                applied_amount=max(0.0, active_applied),
                historical_applied_amount=max(
                    0.0,
                    historical_applied,
                ),
                unapplied_amount=max(0.0, unapplied),

                method=(payment.method or "").strip(),
                reference=(payment.reference or "").strip(),
                note=(payment.note or "").strip(),
                status=status,

                created_by_name=username_for(
                    payment.created_by
                ),
                created_at=format_datetime(
                    payment.created_at
                ),

                posted_by_name=username_for(
                    payment.posted_by
                ),
                posted_at=format_datetime(
                    payment.posted_at
                ),

                voided_by_name=username_for(
                    payment.voided_by
                ),
                voided_at=format_datetime(
                    payment.voided_at
                ),
                void_reason=(
                    payment.void_reason or ""
                ).strip(),

                allocations=allocation_rows,
            )
        )

    return rows

def get_technician_ledger(
    technician_name: str,
) -> list[TechnicianLedgerRow]:
    from models import TechnicianLedgerEntry, User, utc_to_local

    tech = (technician_name or "").strip()

    if not tech:
        return []

    entries = (
        TechnicianLedgerEntry.query
        .filter(
            TechnicianLedgerEntry.technician_name == tech,
            TechnicianLedgerEntry.voided == False,
        )
        .order_by(
            TechnicianLedgerEntry.entry_date.desc(),
            TechnicianLedgerEntry.created_at.desc(),
            TechnicianLedgerEntry.id.desc(),
        )
        .all()
    )

    creator_ids = {
        int(entry.created_by)
        for entry in entries
        if entry.created_by
    }

    users_by_id = {}

    if creator_ids:
        users = (
            User.query
            .filter(User.id.in_(creator_ids))
            .all()
        )

        users_by_id = {
            int(user.id): user.username
            for user in users
        }

    rows = []

    for entry in entries:
        created_by_name = ""

        if entry.created_by:
            created_by_name = users_by_id.get(
                int(entry.created_by),
                f"User #{entry.created_by}",
            )

        created_at_local = utc_to_local(entry.created_at)

        rows.append(
            TechnicianLedgerRow(
                id=entry.id,

                entry_date=(
                    entry.entry_date.strftime("%m/%d/%Y")
                    if entry.entry_date
                    else ""
                ),

                entry_type=entry.entry_type or "",

                supplier_name=entry.supplier_name,
                supplier_invoice=entry.supplier_invoice,
                job_number=entry.job_number,

                amount=round(
                    float(entry.amount or 0.0),
                    2,
                ),
                paid_amount=round(
                    float(entry.paid_amount or 0.0),
                    2,
                ),
                remaining_amount=round(
                    float(entry.remaining_amount or 0.0),
                    2,
                ),

                status=entry.status or "",
                note=entry.note,

                created_by_name=created_by_name,
                created_at=(
                    created_at_local.strftime(
                        "%m/%d/%Y %I:%M %p"
                    )
                    if created_at_local
                    else ""
                ),
            )
        )

    return rows

def get_technician_summary(
    technician_name: str,
) -> TechnicianSummary:
    from models import TechnicianLedgerEntry

    tech = (technician_name or "").strip()

    if not tech:
        raise ValueError("Technician name required")

    rows = (
        TechnicianLedgerEntry.query
        .filter(
            TechnicianLedgerEntry.technician_name == tech,
            TechnicianLedgerEntry.voided == False,
        )
        .all()
    )
    payment_total = (
        db.session.query(
            func.coalesce(
                func.sum(TechnicianPayment.amount),
                0.0,
            )
        )
        .filter(
            TechnicianPayment.technician_name == tech,
            TechnicianPayment.status == "posted",
            TechnicianPayment.voided == False,
        )
        .scalar()
        or 0.0
    )

    charges = 0.0
    payments = 0.0
    adjustments = 0.0
    opening = 0.0

    invoices = set()

    remaining = 0.0

    for r in rows:

        remaining += float(r.remaining_amount or 0)

        if r.supplier_invoice:
            invoices.add(r.supplier_invoice)

        t = (r.entry_type or "").upper()

        if t == "CHARGE":
            charges += float(r.amount or 0)

        elif t == "OPENING_BALANCE":
            opening += float(r.amount or 0)

        elif t in (
            "ADJUSTMENT_PLUS",
            "ADJUSTMENT_MINUS",
        ):
            adjustments += float(r.amount or 0)

        payments = float(payment_total or 0.0)

    return TechnicianSummary(
        technician=tech,

        charges=round(charges, 2),

        payments=round(payments, 2),

        adjustments=round(adjustments, 2),

        opening_balance=round(opening, 2),

        remaining=round(remaining, 2),

        open_invoices=len(invoices),
    )

@dataclass(frozen=True)
class PaymentPreviewLine:
    ledger_entry_id: int
    entry_date: str
    supplier_invoice: str
    job_number: str
    remaining_before: float
    apply_amount: float
    remaining_after: float


@dataclass(frozen=True)
class PaymentPreview:
    technician: str
    payment_amount: float
    total_applied: float
    unapplied_amount: float
    lines: list[PaymentPreviewLine]


def preview_technician_payment_fifo(
    technician_name: str,
    amount: float,
) -> PaymentPreview:
    from models import TechnicianLedgerEntry

    tech = (technician_name or "").strip()
    payment_amount = round(float(amount or 0.0), 2)

    if not tech:
        raise ValueError("Technician name is required")

    if payment_amount <= 0:
        raise ValueError("Payment amount must be greater than zero")

    remaining_payment = payment_amount
    preview_lines = []

    open_entries = (
        TechnicianLedgerEntry.query
        .filter(
            TechnicianLedgerEntry.technician_name == tech,
            TechnicianLedgerEntry.status.in_(["open", "partial"]),
            TechnicianLedgerEntry.remaining_amount > 0,
            TechnicianLedgerEntry.voided == False,
        )
        .order_by(
            TechnicianLedgerEntry.entry_date.asc(),
            TechnicianLedgerEntry.created_at.asc(),
            TechnicianLedgerEntry.id.asc(),
        )
        .all()
    )

    for entry in open_entries:
        if remaining_payment <= 0:
            break

        entry_remaining = round(float(entry.remaining_amount or 0.0), 2)
        if entry_remaining <= 0:
            continue

        apply_amount = round(min(remaining_payment, entry_remaining), 2)
        remaining_after = round(entry_remaining - apply_amount, 2)

        preview_lines.append(
            PaymentPreviewLine(
                ledger_entry_id=entry.id,
                entry_date=entry.entry_date.strftime("%m/%d/%Y") if entry.entry_date else "",
                supplier_invoice=entry.supplier_invoice or "",
                job_number=entry.job_number or "",
                remaining_before=entry_remaining,
                apply_amount=apply_amount,
                remaining_after=remaining_after,
            )
        )

        remaining_payment = round(remaining_payment - apply_amount, 2)

    total_applied = round(payment_amount - remaining_payment, 2)

    return PaymentPreview(
        technician=tech,
        payment_amount=payment_amount,
        total_applied=total_applied,
        unapplied_amount=round(remaining_payment, 2),
        lines=preview_lines,
    )