from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import func

from models import (
    GoodsReceipt,
    GoodsReceiptLine,
    IssuedPartRecord,
    ReturnDestination,
    WorkOrder,
    WorkOrderPart,
)


# ============================================================
# DTO
# ============================================================

@dataclass(slots=True)
class ReturnCandidate:
    record: IssuedPartRecord

    amount: float

    technician: str

    job: str

    label: str


@dataclass(slots=True)
class InvoiceCandidate:
    receipt_line: GoodsReceiptLine

    receipt: GoodsReceipt

    work_order: WorkOrder | None

    work_order_part: WorkOrderPart | None

    amount: float

    technician: str

    job: str

    invoice_number: str

    label: str


# ============================================================
# COMMON
# ============================================================

def validate_component_total(
    statement_amount: float,
    component_amounts: list[float],
):
    statement_amount = round(
        abs(float(statement_amount)),
        2,
    )

    entered_total = round(
        sum(component_amounts),
        2,
    )

    difference = round(
        statement_amount - entered_total,
        2,
    )

    if abs(difference) > 0.009:
        raise ValueError(
            "Component amounts must equal "
            f"${statement_amount:,.2f}. "
            f"Difference ${difference:,.2f}"
        )


# ============================================================
# RETURNS
# ============================================================

def build_return_label(
    record: IssuedPartRecord,
) -> str:

    technician = (
        record.issued_to or ""
    ).strip()

    job = (
        record.reference_job or ""
    ).replace(
        "RETURN ",
        "",
    ).strip()

    amount = round(
        abs(
            float(record.quantity)
            * float(record.unit_cost_at_issue)
        ),
        2,
    )

    if technician and job:
        return (
            f"${amount:,.2f}"
            f" • {technician}"
            f" / {job}"
        )

    return f"${amount:,.2f}"


def find_return_candidates(
    *,
    supplier_name: str,
    amount: float,
    excluded_ids: set[int] | None = None,
):

    query = (
        IssuedPartRecord.query
        .join(
            ReturnDestination,
            IssuedPartRecord.return_destination_id
            == ReturnDestination.id,
        )
        .filter(

            func.upper(
                func.trim(
                    IssuedPartRecord.return_to
                )
            ) == "VENDOR",

            IssuedPartRecord.quantity < 0,

            func.lower(
                func.trim(
                    ReturnDestination.name
                )
            )
            == supplier_name.lower(),

            func.abs(
                func.abs(
                    IssuedPartRecord.quantity
                    * IssuedPartRecord.unit_cost_at_issue
                )
                - float(amount)
            )
            <= 0.011,
        )
    )

    if excluded_ids:

        query = query.filter(
            ~IssuedPartRecord.id.in_(
                excluded_ids
            )
        )

    candidates = []

    for record in query.all():

        candidates.append(

            ReturnCandidate(
                record=record,

                amount=round(
                    abs(
                        float(record.quantity)
                        * float(
                            record.unit_cost_at_issue
                        )
                    ),
                    2,
                ),

                technician=(
                    record.issued_to or ""
                ).strip(),

                job=(
                    record.reference_job or ""
                ).replace(
                    "RETURN ",
                    "",
                ).strip(),

                label=build_return_label(
                    record
                ),
            )
        )

    return candidates


# ============================================================
# INVOICES
# ============================================================

def build_invoice_label(
    *,
    technician,
    job,
    invoice_number,
    amount,
):

    return (
        f"${amount:,.2f}"
        f" • {technician}"
        f" / {job}"
        f" • INV {invoice_number}"
    )


def find_invoice_candidates(
    *,
    supplier_name: str,
    amount: float,
    excluded_receipt_lines: set[int] | None = None,
):
    candidates = []

    query = (
        GoodsReceiptLine.query
        .join(
            GoodsReceipt,
            GoodsReceipt.id
            == GoodsReceiptLine.goods_receipt_id,
        )
        .filter(
            func.lower(
                func.trim(
                    GoodsReceipt.supplier_name
                )
            )
            == supplier_name.strip().lower()
        )
    )

    if excluded_receipt_lines:
        query = query.filter(
            ~GoodsReceiptLine.id.in_(
                excluded_receipt_lines
            )
        )

    receipt_lines = (
        query
        .order_by(
            GoodsReceipt.invoice_date.desc(),
            GoodsReceipt.id.desc(),
            GoodsReceiptLine.id.desc(),
        )
        .all()
    )

    for receipt_line in receipt_lines:

        qty = abs(
            float(
                receipt_line.quantity or 0
            )
        )

        if (
            receipt_line.actual_unit_cost
            is not None
        ):
            unit_cost = (
                receipt_line.actual_unit_cost
            )
        else:
            unit_cost = (
                receipt_line.unit_cost
            )

        total = round(
            qty
            * abs(float(unit_cost)),
            2,
        )

        if abs(total - amount) > 0.01:
            continue

        issued_record = (
            IssuedPartRecord.query
            .filter(
                IssuedPartRecord.source_receipt_line_id
                == receipt_line.id,

                IssuedPartRecord.quantity > 0,
            )
            .order_by(
                IssuedPartRecord.issue_date.desc(),
                IssuedPartRecord.id.desc(),
            )
            .first()
        )

        work_order_part = None
        work_order = None

        technician = ""
        job = ""

        if issued_record:
            technician = (
                issued_record.issued_to or ""
            ).strip()

            job = (
                issued_record.reference_job or ""
            ).strip()

            if (
                issued_record.batch
                and issued_record.batch.work_order_id
            ):
                work_order = WorkOrder.query.get(
                    issued_record.batch.work_order_id
                )

                if work_order:
                    technician = (
                        work_order.technician_name
                        or technician
                    ).strip()

                    job = (
                        work_order.job_numbers
                        or job
                    ).strip()

        candidates.append(

            InvoiceCandidate(

                receipt_line=receipt_line,

                receipt=receipt_line.goods_receipt,

                work_order=work_order,

                work_order_part=work_order_part,

                amount=total,

                technician=technician,

                job=job,

                invoice_number=(
                    receipt_line.goods_receipt.invoice_number
                ),

                label=build_invoice_label(

                    technician=technician,

                    job=job,

                    invoice_number=(
                        receipt_line.goods_receipt.invoice_number
                    ),

                    amount=total,
                ),
            )
        )

    return candidates