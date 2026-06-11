from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from sqlalchemy import func
from extensions import db
from models import IssuedPartRecord, IssuedBatch, Part


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
        payment = TechnicianPayment(
            technician_name=tech,
            payment_date=payment_date,
            amount=pay_amount,
            unapplied_amount=pay_amount,
            method=(method or "").strip() or None,
            reference=(reference or "").strip() or None,
            note=(note or "").strip() or None,
            created_by=created_by,
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

def create_ledger_charge_from_issued_record(
    record,
    invoice_number: int | str | None = None,
    work_order_id: int | None = None,
    default_technician: str | None = None,
    default_job: str | None = None,
    note_prefix: str | None = None,
):
    """
    Create CHARGE ledger entry from IssuedPartRecord.
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

    existing = TechnicianLedgerEntry.query.filter_by(
        issued_part_record_id=record.id,
        entry_type="CHARGE",
    ).first()

    if existing:
        existing_amount = round(float(existing.amount or 0.0), 2)
        new_amount = round(float(amount or 0.0), 2)

        existing.supplier_invoice = str(supplier_invoice).strip() if supplier_invoice else existing.supplier_invoice
        existing.job_number = str(job_number).strip() if job_number else existing.job_number
        existing.note = note[:500]

        # If no payment was applied yet, safely refresh the charge amount
        if float(existing.paid_amount or 0.0) <= 0:
            existing.amount = new_amount
            existing.remaining_amount = new_amount
            existing.status = "open" if new_amount > 0 else "paid"

        # If payment already exists, do not rewrite accounting history
        return existing

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