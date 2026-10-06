from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from flask import flash

from extensions import db

from models import (
    GoodsReceipt,
    GoodsReceiptLine,
    IssuedPartRecord,
    SupplierStatementLine,
    SupplierStatementInvoiceComponent,
    WorkOrder,
    WorkOrderPart,
)

from services.statement_matching_helper import (
    find_invoice_candidates,
    validate_component_total,
)


@dataclass(slots=True)
class InvoiceMatchPage:

    line: SupplierStatementLine

    candidate_map: dict[
        int,
        list,
    ]


def _used_receipt_lines(
    line_id: int,
) -> set[int]:

    return {
        int(receipt_line_id)
        for (receipt_line_id,) in (
            db.session.query(
                SupplierStatementInvoiceComponent
                .matched_goods_receipt_line_id
            )
            .filter(
                SupplierStatementInvoiceComponent
                .matched_goods_receipt_line_id
                .isnot(None),

                SupplierStatementInvoiceComponent
                .statement_line_id != line_id,
            )
            .all()
        )
        if receipt_line_id is not None
    }


def _load_line(
    line_id: int,
):

    return (
        SupplierStatementLine.query
        .get_or_404(line_id)
    )


def load_page(
    line_id: int,
) -> InvoiceMatchPage:

    line = _load_line(
        line_id,
    )

    supplier_name = (
        line.supplier_name
        or line.statement.supplier_name
        or ""
    ).strip()

    used = _used_receipt_lines(
        line.id,
    )

    candidate_map = {}

    for component in (
        line.invoice_components
        or []
    ):

        candidates = (
            find_invoice_candidates(
                supplier_name=supplier_name,
                amount=float(
                    component.amount or 0
                ),
                excluded_receipt_lines=used,
            )
        )

        candidate_map[
            component.id
        ] = candidates

    return InvoiceMatchPage(
        line=line,
        candidate_map=candidate_map,
    )

def save_components(
    *,
    line_id: int,
    form,
    current_user,
):

    line = _load_line(
        line_id,
    )

    supplier_name = (
        line.supplier_name
        or line.statement.supplier_name
        or ""
    ).strip()

    amount_values = form.getlist(
        "component_amount"
    )

    selected_values = form.getlist(
        "component_receipt_line_id"
    )

    amounts = []

    for raw in amount_values:

        raw = (
            raw or ""
        ).strip()

        if not raw:
            continue

        amount = round(
            float(raw),
            2,
        )

        if amount <= 0:
            raise ValueError(
                "Every invoice amount must be greater than zero."
            )

        amounts.append(amount)

    if not amounts:
        raise ValueError(
            "Add at least one invoice amount."
        )

    statement_amount = round(
        abs(
            float(
                line.invoice_amount
                or line.open_balance
                or 0
            )
        ),
        2,
    )

    validate_component_total(
        statement_amount=statement_amount,
        component_amounts=amounts,
    )

    while (
        len(selected_values)
        < len(amounts)
    ):
        selected_values.append("")

    used_receipt_lines = (
        _used_receipt_lines(
            line.id,
        )
    )

    for component in list(
        line.invoice_components
        or []
    ):
        db.session.delete(
            component
        )

    db.session.flush()

    selected_in_this_split = set()

    for index, amount in enumerate(
        amounts
    ):

        selected_raw = (
            selected_values[index]
            or ""
        ).strip()

        excluded = (
            used_receipt_lines
            | selected_in_this_split
        )

        candidates = (
            find_invoice_candidates(
                supplier_name=supplier_name,
                amount=amount,
                excluded_receipt_lines=excluded,
            )
        )

        candidate_ids = {
            c.receipt_line.id
            for c in candidates
        }

        matched_receipt_line_id = None

        note = None

        if selected_raw:

            selected_id = int(
                selected_raw
            )

            if (
                selected_id
                not in candidate_ids
            ):
                raise ValueError(
                    "Selected invoice match is no longer available."
                )

            matched_receipt_line_id = (
                selected_id
            )

            selected_in_this_split.add(
                selected_id
            )

        elif len(candidates) == 1:

            matched_receipt_line_id = (
                candidates[0]
                .receipt_line
                .id
            )

            selected_in_this_split.add(
                matched_receipt_line_id
            )

        elif len(candidates) == 0:

            note = (
                "No matching receipt line was found."
            )

        else:

            note = (
                f"{len(candidates)} possible matches were found."
            )

        db.session.add(
            SupplierStatementInvoiceComponent(

                statement_line_id=line.id,

                amount=amount,

                matched_goods_receipt_line_id=(
                    matched_receipt_line_id
                ),

                note=note,

                created_by=getattr(
                    current_user,
                    "id",
                    None,
                ),
            )
        )

    db.session.commit()

    db.session.refresh(
        line,
    )

    matched = sum(
        1
        for c in (
            line.invoice_components
            or []
        )
        if c.matched_goods_receipt_line_id
    )

    total = len(
        line.invoice_components
        or []
    )

    if matched == total:

        flash(
            f"Invoice split saved. "
            f"All {matched} components matched.",
            "success",
        )

    else:

        flash(
            f"Invoice split saved. "
            f"{matched} of {total} matched.",
            "warning",
        )