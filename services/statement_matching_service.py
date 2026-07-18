from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from sqlalchemy import func

from models import (
    SupplierStatement,
    SupplierStatementLine,
    SupplierReturnBatch,
)


@dataclass(slots=True)
class StatementMatchRow:

    statement_line: SupplierStatementLine

    matched_batch: Optional[SupplierReturnBatch] = None

    technician_job: str = ""

    system_amount: float = 0.0

    difference: float = 0.0

    status: str = "NOT_FOUND"


@dataclass(slots=True)
class StatementMatchResult:

    statement: SupplierStatement

    rows: list[StatementMatchRow] = field(default_factory=list)

    matched_count: int = 0

    not_found_count: int = 0

    already_reconciled_count: int = 0


def build_statement_view(
    statement_id: int,
) -> StatementMatchResult:

    statement = SupplierStatement.query.get_or_404(statement_id)

    result = StatementMatchResult(statement=statement)

    for line in statement.lines:

        row = StatementMatchRow(
            statement_line=line,
        )

        matches = (
            SupplierReturnBatch.query
            .filter(
                SupplierReturnBatch.status == "posted",
                SupplierReturnBatch.supplier_name == line.supplier_name,
                func.abs(
                    SupplierReturnBatch.total_value -
                    float(line.credit_amount or 0)
                ) <= 0.01,
            )
            .all()
        )

        if len(matches) == 0:

            row.status = "NOT_FOUND"
            result.not_found_count += 1

        elif len(matches) == 1:

            batch = matches[0]

            row.matched_batch = batch
            row.system_amount = float(batch.total_value or 0)

            row.difference = round(
                float(line.credit_amount or 0) -
                row.system_amount,
                2,
            )

            jobs = []

            for item in batch.items:
                if item.tech_note:
                    jobs.append(item.tech_note)

            row.technician_job = ", ".join(sorted(set(jobs)))

            if batch.reconciled_at:
                row.status = "ALREADY_RECONCILED"
                result.already_reconciled_count += 1
            else:
                row.status = "MATCHED"
                result.matched_count += 1

        else:

            row.status = "MULTIPLE_MATCHES"

        result.rows.append(row)

    return result