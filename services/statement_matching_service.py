from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from sqlalchemy import func

from models import (
    GoodsReceipt,
    IssuedPartRecord,
    ReturnDestination,
    SupplierStatement,
    SupplierStatementLine,
    WorkOrder,
    WorkOrderPart,
)


@dataclass(slots=True)
class StatementMatchRow:
    statement_line: SupplierStatementLine

    # Returns
    matched_record: Optional[IssuedPartRecord] = None

    # Open invoices
    matched_work_order_id: Optional[int] = None
    matched_receiving_id: Optional[int] = None

    technician_job: str = ""
    match_label: str = ""

    system_amount: float = 0.0
    difference: float = 0.0

    status: str = "NOT_FOUND"


@dataclass(slots=True)
class StatementMatchResult:
    statement: SupplierStatement
    rows: list[StatementMatchRow] = field(default_factory=list)

    matched_count: int = 0
    multiple_count: int = 0
    not_found_count: int = 0
    already_reconciled_count: int = 0


# ============================================================
# COMMON HELPERS
# ============================================================

def _normalize_text(value: object) -> str:
    return str(value or "").strip()

def _supplier_aliases(value: object) -> set[str]:
    supplier = (
        str(value or "")
        .strip()
        .lower()
    )

    aliases = {supplier}

    if supplier in {
        "reliable parts",
        "reliable parts inc",
        "reliable",
        "rel",
    }:
        aliases.update({
            "reliable parts",
            "reliable parts inc",
            "reliable",
            "rel",
        })

    if supplier in {
        "marcone",
        "mar",
    }:
        aliases.update({
            "marcone",
            "mar",
        })

    return {
        alias
        for alias in aliases
        if alias
    }


def _normalize_invoice_number(value: object) -> str:
    return _normalize_text(value)


def _clean_job_number(value: str | None) -> str:
    job = (value or "").strip()

    if job.upper().startswith("RETURN "):
        job = job[7:].strip()

    return job


# ============================================================
# RETURN MATCHING HELPERS
# ============================================================

def _build_technician_job(
    record: IssuedPartRecord,
) -> str:
    technician = (
        record.issued_to or ""
    ).strip()

    job_number = _clean_job_number(
        record.reference_job
    )

    amount = round(
        abs(
            float(record.quantity or 0)
            * float(
                record.unit_cost_at_issue or 0.0
            )
        ),
        2,
    )

    label = f"RETURN • ${amount:,.2f}"

    if technician and job_number:
        return (
            f"{label} - "
            f"{technician} / {job_number}"
        )

    if technician:
        return f"{label} - {technician}"

    if job_number:
        return f"{label} - {job_number}"

    return label


# ============================================================
# OPEN INVOICE — RECEIVING AMOUNT
# ============================================================

def _receipt_lines(receipt: GoodsReceipt) -> list:
    """
    Supports either relationship name:
        receipt.lines
        receipt.items
    """

    lines = getattr(
        receipt,
        "lines",
        None,
    )

    if lines is not None:
        return list(lines or [])

    items = getattr(
        receipt,
        "items",
        None,
    )

    return list(items or [])


def _receipt_line_quantity(line) -> float:
    for field_name in (
        "quantity",
        "qty",
        "received_qty",
    ):
        value = getattr(
            line,
            field_name,
            None,
        )

        if value is not None:
            return abs(float(value or 0.0))

    return 0.0


def _receipt_line_final_unit_cost(line) -> float:
    """
    Final supplier unit cost including allocated delivery/fees.

    For the example invoice 91948923 this must return 117.48,
    not the base cost 107.99.
    """

    for field_name in (
        "actual_unit_cost",
        "adjusted_unit_cost",
        "final_unit_cost",
        "unit_cost",
    ):
        value = getattr(
            line,
            field_name,
            None,
        )

        if value is not None:
            return abs(float(value or 0.0))

    base_cost = None

    for field_name in (
        "base_unit_cost",
        "unit_price_base",
        "base_cost",
    ):
        value = getattr(
            line,
            field_name,
            None,
        )

        if value is not None:
            base_cost = abs(float(value or 0.0))
            break

    if base_cost is None:
        base_cost = 0.0

    allocated_fee = 0.0

    for field_name in (
        "extra_alloc_per_unit",
        "allocated_fee",
        "fee_per_unit",
        "delivery_per_unit",
    ):
        value = getattr(
            line,
            field_name,
            None,
        )

        if value is not None:
            allocated_fee = abs(
                float(value or 0.0)
            )
            break

    return base_cost + allocated_fee


def _receipt_lines_total(
    receipt: GoodsReceipt,
) -> float:
    total = 0.0

    for line in _receipt_lines(receipt):
        quantity = _receipt_line_quantity(
            line
        )

        unit_cost = (
            _receipt_line_final_unit_cost(
                line
            )
        )

        total += quantity * unit_cost

    return round(total, 2)


def _receipt_total(
    receipt: GoodsReceipt,
) -> float:
    """
    Use the final posted Receiving total when available.

    Do not add extra_expenses again. The posted receipt total or
    adjusted line cost already contains allocated delivery/fees.
    """

    for field_name in (
        "total_cost",
        "total_value",
        "grand_total",
        "invoice_total",
    ):
        value = getattr(
            receipt,
            field_name,
            None,
        )

        if value is not None:
            amount = abs(float(value or 0.0))

            if amount > 0:
                return round(amount, 2)

    return _receipt_lines_total(
        receipt
    )


def _find_invoice_receipts(
    *,
    supplier_name: str,
    document_number: str,
) -> list[GoodsReceipt]:
    supplier = (
        supplier_name or ""
    ).strip().lower()

    document = _normalize_invoice_number(
        document_number
    )

    if not document:
        return []

    query = GoodsReceipt.query.filter(
        func.trim(
            func.coalesce(
                GoodsReceipt.invoice_number,
                "",
            )
        ) == document
    )

    if hasattr(
        GoodsReceipt,
        "supplier_name",
    ):
        query = query.filter(
            func.lower(
                func.trim(
                    func.coalesce(
                        GoodsReceipt.supplier_name,
                        "",
                    )
                )
            ) == supplier
        )

    return (
        query
        .order_by(
            GoodsReceipt.id.asc()
        )
        .all()
    )


def _select_invoice_receipts(
    *,
    receipts: list[GoodsReceipt],
    statement_amount: float,
) -> tuple[
    list[GoodsReceipt],
    float,
    str,
]:
    """
    Returns:
        selected receipts
        calculated total
        status

    Status:
        MATCHED
        MISMATCH
        MULTIPLE
        NOT_FOUND
    """

    if not receipts:
        return [], 0.0, "NOT_FOUND"

    receipt_totals = [
        (
            receipt,
            _receipt_total(receipt),
        )
        for receipt in receipts
    ]

    exact_receipts = [
        (
            receipt,
            total,
        )
        for receipt, total in receipt_totals
        if abs(
            statement_amount - total
        ) <= 0.01
    ]

    # Exactly one receipt has the correct amount.
    if len(exact_receipts) == 1:
        receipt, total = exact_receipts[0]

        return (
            [receipt],
            total,
            "MATCHED",
        )

    # More than one exact candidate.
    if len(exact_receipts) > 1:
        return [], 0.0, "MULTIPLE"

    # Only one receipt exists, but amount differs.
    if len(receipt_totals) == 1:
        receipt, total = receipt_totals[0]

        return (
            [receipt],
            total,
            "MISMATCH",
        )

    # Consolidated supplier invoice may consist of several
    # Receiving batches with the same invoice number.
    combined_total = round(
        sum(
            total
            for receipt, total
            in receipt_totals
        ),
        2,
    )

    if abs(
        statement_amount - combined_total
    ) <= 0.01:
        return (
            [
                receipt
                for receipt, total
                in receipt_totals
            ],
            combined_total,
            "MATCHED",
        )

    return (
        [],
        combined_total,
        "MULTIPLE",
    )


# ============================================================
# OPEN INVOICE — TECHNICIAN / JOB MATCH
# ============================================================

def _find_invoice_work_order_rows(
    document_number: str,
) -> list[tuple[WorkOrderPart, WorkOrder]]:
    document = _normalize_invoice_number(
        document_number
    )

    if not document:
        return []

    return (
        WorkOrderPart.query
        .with_entities(
            WorkOrderPart,
            WorkOrder,
        )
        .join(
            WorkOrder,
            WorkOrder.id
            == WorkOrderPart.work_order_id,
        )
        .filter(
            func.trim(
                func.coalesce(
                    WorkOrderPart.invoice_number,
                    "",
                )
            ) == document,
        )
        .order_by(
            WorkOrder.id.asc(),
            WorkOrderPart.id.asc(),
        )
        .all()
    )


def _work_order_technician(
    work_order: WorkOrder,
) -> str:
    for field_name in (
        "technician_name",
        "technician_username",
        "technician",
        "assigned_to",
    ):
        value = getattr(
            work_order,
            field_name,
            None,
        )

        if value:
            return str(value).strip()

    return "NO TECH"


def _work_order_jobs(
    work_order: WorkOrder,
) -> str:
    for field_name in (
        "job_numbers",
        "job_number",
        "reference_job",
        "canonical_ref",
    ):
        value = getattr(
            work_order,
            field_name,
            None,
        )

        if value:
            return str(value).strip()

    return f"WO #{work_order.id}"


def _invoice_work_order_label(
    rows: list[
        tuple[WorkOrderPart, WorkOrder]
    ],
) -> str:
    labels: list[str] = []
    seen_work_orders: set[int] = set()

    for part, work_order in rows:
        if work_order.id in seen_work_orders:
            continue

        seen_work_orders.add(
            work_order.id
        )

        technician = _work_order_technician(
            work_order
        )

        jobs = _work_order_jobs(
            work_order
        )

        labels.append(
            f"{technician} / {jobs}"
        )

    return "\n".join(labels)


# ============================================================
# MAIN STATEMENT VIEW
# ============================================================

def build_statement_view(
    statement_id: int,
) -> StatementMatchResult:
    statement = (
        SupplierStatement.query
        .get_or_404(statement_id)
    )

    result = StatementMatchResult(
        statement=statement
    )

    for line in statement.lines:
        row = StatementMatchRow(
            statement_line=line
        )

        line_type = (
            line.line_type or ""
        ).strip().lower()

        supplier_name = (
            line.supplier_name
            or statement.supplier_name
            or ""
        ).strip()

        # ====================================================
        # OPEN INVOICES
        #
        # Invoice amount comes from Receiving.
        # Technician/job comes from WorkOrderPart.
        # ====================================================
        if line_type == "invoice":
            document_number = (
                line.document_number or ""
            ).strip()

            statement_amount = round(
                abs(
                    float(
                        line.invoice_amount
                        or line.open_balance
                        or 0.0
                    )
                ),
                2,
            )

            # ================================================
            # MANUAL INVOICE MATCH COMPONENTS
            # Saved from the Invoice Match page.
            # ================================================
            invoice_components = list(
                line.invoice_components or []
            )

            if invoice_components:
                components_total = round(
                    sum(
                        float(component.amount or 0.0)
                        for component in invoice_components
                    ),
                    2,
                )

                row.system_amount = components_total

                row.difference = round(
                    statement_amount - components_total,
                    2,
                )

                labels = []
                matched_receiving_ids = []
                matched_work_order_ids = []

                all_matched = True

                for component in invoice_components:
                    receipt_line = (
                        component.matched_receipt_line
                    )

                    if receipt_line is None:
                        all_matched = False

                        labels.append(
                            f"INVOICE • "
                            f"${float(component.amount or 0.0):,.2f}"
                            f" • Unmatched"
                        )
                        continue

                    receipt = receipt_line.goods_receipt

                    if receipt is not None:
                        matched_receiving_ids.append(
                            receipt.id
                        )

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

                    technician = ""
                    job_number = ""
                    work_order = None

                    if issued_record is not None:
                        technician = (
                            issued_record.issued_to or ""
                        ).strip()

                        job_number = (
                            issued_record.reference_job or ""
                        ).strip()

                        if (
                            issued_record.batch
                            and issued_record.batch.work_order_id
                        ):
                            work_order = WorkOrder.query.get(
                                issued_record.batch.work_order_id
                            )

                    if work_order is not None:
                        matched_work_order_ids.append(
                            work_order.id
                        )

                        technician = (
                            work_order.technician_name
                            or technician
                        ).strip()

                        job_number = (
                            work_order.job_numbers
                            or job_number
                        ).strip()

                    invoice_number = ""

                    if receipt is not None:
                        invoice_number = str(
                            receipt.invoice_number or ""
                        ).strip()

                    label = (
                        f"INVOICE • "
                        f"${float(component.amount or 0.0):,.2f}"
                    )

                    if technician and job_number:
                        label += (
                            f" • {technician} / {job_number}"
                        )
                    elif technician:
                        label += f" • {technician}"
                    elif job_number:
                        label += f" • {job_number}"

                    if invoice_number:
                        label += f" • INV {invoice_number}"

                    labels.append(label)

                row.match_label = "\n".join(labels)

                if matched_receiving_ids:
                    row.matched_receiving_id = (
                        matched_receiving_ids[0]
                    )

                if matched_work_order_ids:
                    row.matched_work_order_id = (
                        matched_work_order_ids[0]
                    )

                if (
                    all_matched
                    and abs(row.difference) <= 0.009
                ):
                    row.status = "INVOICE_MATCHED"
                    result.matched_count += 1

                elif abs(row.difference) > 0.009:
                    row.status = (
                        "INVOICE_AMOUNT_MISMATCH"
                    )
                    result.not_found_count += 1

                else:
                    row.status = "INVOICE_NOT_FOUND"
                    result.not_found_count += 1

                result.rows.append(row)
                continue

            # ================================================
            # AUTOMATIC MATCH
            # Used only when no manual components exist.
            # ================================================
            receipts = _find_invoice_receipts(
                supplier_name=supplier_name,
                document_number=document_number,
            )

            (
                selected_receipts,
                system_amount,
                invoice_status,
            ) = _select_invoice_receipts(
                receipts=receipts,
                statement_amount=statement_amount,
            )

            work_order_rows = (
                _find_invoice_work_order_rows(
                    document_number
                )
            )

            row.system_amount = round(
                system_amount,
                2,
            )

            row.difference = round(
                statement_amount
                - row.system_amount,
                2,
            )

            row.match_label = (
                _invoice_work_order_label(
                    work_order_rows
                )
            )

            work_order_ids = sorted(
                {
                    work_order.id
                    for part, work_order
                    in work_order_rows
                }
            )

            if work_order_ids:
                row.matched_work_order_id = (
                    work_order_ids[0]
                )

            if selected_receipts:
                row.matched_receiving_id = (
                    selected_receipts[0].id
                )

            if invoice_status == "MATCHED":
                row.status = "INVOICE_MATCHED"
                result.matched_count += 1

            elif invoice_status == "MISMATCH":
                row.status = (
                    "INVOICE_AMOUNT_MISMATCH"
                )
                result.not_found_count += 1

            elif invoice_status == "MULTIPLE":
                row.status = "INVOICE_MULTIPLE"
                result.multiple_count += 1

            else:
                row.status = "INVOICE_NOT_FOUND"
                result.not_found_count += 1

            result.rows.append(row)
            continue

        # ====================================================
        # MANUAL SPLIT FOR RETURNS / CREDITS
        # ====================================================
        components = list(
            line.components or []
        )

        if components:
            components_total = round(
                sum(
                    float(
                        component.amount or 0.0
                    )
                    for component in components
                ),
                2,
            )

            statement_amount = round(
                abs(
                    float(
                        line.credit_amount or 0.0
                    )
                ),
                2,
            )

            row.system_amount = (
                components_total
            )

            row.difference = round(
                statement_amount
                - components_total,
                2,
            )

            row.technician_job = "\n".join(
                component.display_match
                for component in components
            )

            all_matched = all(
                component
                .matched_issued_part_record_id
                for component in components
            )

            if (
                all_matched
                and abs(row.difference) <= 0.009
            ):
                row.status = "MATCHED"
                result.matched_count += 1

            else:
                row.status = "PARTIAL_MATCH"
                result.not_found_count += 1

            result.rows.append(row)
            continue

        # ====================================================
        # NORMAL RETURNS / CREDITS
        # ====================================================
        statement_amount = round(
            abs(
                float(
                    line.credit_amount or 0.0
                )
            ),
            2,
        )

        supplier_aliases = _supplier_aliases(
            supplier_name
        )

        matches = (
            IssuedPartRecord.query
            .join(
                ReturnDestination,
                IssuedPartRecord
                .return_destination_id
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
                ).in_(supplier_aliases),

                func.abs(
                    func.abs(
                        IssuedPartRecord.quantity
                        * IssuedPartRecord
                        .unit_cost_at_issue
                    )
                    - statement_amount
                ) <= 0.011,
            )
            .order_by(
                IssuedPartRecord.issue_date.desc(),
                IssuedPartRecord.id.desc(),
            )
            .all()
        )

        if not matches:
            row.status = "NOT_FOUND"
            result.not_found_count += 1

        elif len(matches) == 1:
            record = matches[0]

            row.matched_record = record

            row.system_amount = round(
                abs(
                    float(
                        record.quantity or 0
                    )
                    * float(
                        record.unit_cost_at_issue
                        or 0.0
                    )
                ),
                2,
            )

            row.difference = round(
                statement_amount
                - row.system_amount,
                2,
            )

            row.technician_job = (
                _build_technician_job(
                    record
                )
            )

            row.status = "MATCHED"
            result.matched_count += 1

        else:
            row.status = "MULTIPLE_MATCHES"
            result.multiple_count += 1

        result.rows.append(row)

    return result