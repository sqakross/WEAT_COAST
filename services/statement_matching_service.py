from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from sqlalchemy import func

from models import (
    SupplierStatement,
    SupplierStatementLine,
    IssuedPartRecord,
    ReturnDestination,
)


@dataclass(slots=True)
class StatementMatchRow:
    statement_line: SupplierStatementLine

    matched_record: Optional[IssuedPartRecord] = None

    technician_job: str = ""
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


def _clean_job_number(value: str | None) -> str:
    job = (value or "").strip()

    if job.upper().startswith("RETURN "):
        job = job[7:].strip()

    return job


def _build_technician_job(record: IssuedPartRecord) -> str:
    technician = (record.issued_to or "").strip()
    job_number = _clean_job_number(record.reference_job)

    if technician and job_number:
        return f"RETURN • {technician} / {job_number}"

    if technician:
        return f"RETURN • {technician}"

    if job_number:
        return f"RETURN • {job_number}"

    return "RETURN"


def build_statement_view(
    statement_id: int,
) -> StatementMatchResult:

    statement = SupplierStatement.query.get_or_404(
        statement_id
    )

    result = StatementMatchResult(
        statement=statement,
    )

    for line in statement.lines:

        row = StatementMatchRow(
            statement_line=line,
        )

        components = list(line.components or [])

        if components:
            components_total = round(
                sum(
                    float(component.amount or 0.0)
                    for component in components
                ),
                2,
            )

            statement_amount = round(
                abs(float(line.credit_amount or 0.0)),
                2,
            )

            row.system_amount = components_total

            row.difference = round(
                statement_amount - components_total,
                2,
            )

            row.technician_job = "\n".join(
                component.display_match
                for component in components
            )

            all_matched = all(
                component.matched_issued_part_record_id
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

        document_number = (
            line.document_number or ""
        ).strip()

        statement_amount = round(
            abs(float(line.credit_amount or 0.0)),
            2,
        )

        supplier_name = (
            line.supplier_name
            or statement.supplier_name
            or ""
        ).strip().lower()

        matches = (
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
                ) == supplier_name,


                func.abs(
                    func.abs(
                        IssuedPartRecord.quantity
                        * IssuedPartRecord.unit_cost_at_issue
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
                    float(record.quantity or 0)
                    * float(record.unit_cost_at_issue or 0.0)
                ),
                2,
            )

            row.difference = round(
                statement_amount - row.system_amount,
                2,
            )

            row.technician_job = (
                _build_technician_job(record)
            )

            row.status = "MATCHED"
            result.matched_count += 1

        else:
            row.status = "MULTIPLE_MATCHES"
            result.multiple_count += 1

        result.rows.append(row)

    return result