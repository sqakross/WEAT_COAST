from __future__ import annotations
import json
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import func

from extensions import db
from models import (
    AssetAssignment,
    AssetAssignmentReceipt,
    Part,
    ToolAsset,
    ToolMovement,
    WorkOrderAudit,
)


class AssetAssignmentError(Exception):
    """Expected business error during asset assignment."""


@dataclass
class AssetAssignmentResult:
    asset: ToolAsset
    assignment: AssetAssignment
    receipt: AssetAssignmentReceipt
    movement: ToolMovement


def _next_receipt_number() -> str:
    """
    Reserve the next human-readable asset receipt number.

    SQLite allows only one writer at a time, but the unique constraint on
    receipt_number remains the final protection against duplicates.
    """
    next_id = (
        db.session.query(
            func.coalesce(
                func.max(AssetAssignmentReceipt.id),
                0,
            )
        )
        .scalar()
        or 0
    ) + 1

    return f"AR-{int(next_id):06d}"


def _find_inventory_part(
    *,
    part_number: str,
    warehouse: str | None = None,
) -> Part | None:
    query = Part.query.filter(
        func.upper(Part.part_number) == part_number.upper()
    )

    rows = query.all()
    if not rows:
        return None

    warehouse_normalized = (warehouse or "").strip().lower()

    if warehouse_normalized:
        for row in rows:
            row_location = (
                getattr(row, "location", "") or ""
            ).strip().lower()

            if row_location == warehouse_normalized:
                return row

    return rows[0]


def _calculate_free_quantity(asset: ToolAsset) -> int:
    """
    Compatibility calculation for aggregate ToolAsset records.

    Current active quantity:
        assigned movements - returned movements

    AssetAssignment will become the source of truth after all old
    assignments have been migrated.
    """
    assigned_quantity = (
        db.session.query(
            func.coalesce(
                func.sum(ToolMovement.quantity),
                0,
            )
        )
        .filter(
            ToolMovement.tool_id == asset.id,
            ToolMovement.action == "assigned",
        )
        .scalar()
        or 0
    )

    returned_quantity = (
        db.session.query(
            func.coalesce(
                func.sum(ToolMovement.quantity),
                0,
            )
        )
        .filter(
            ToolMovement.tool_id == asset.id,
            ToolMovement.action == "returned",
        )
        .scalar()
        or 0
    )

    active_quantity = max(
        int(assigned_quantity) - int(returned_quantity),
        0,
    )

    total_quantity = max(int(asset.quantity or 0), 0)

    return max(total_quantity - active_quantity, 0)


def assign_tool_from_work_order(
    *,
    work_order,
    line,
    current_user,
) -> AssetAssignmentResult:
    """
    Assign one reusable asset from a Work Order.

    Creates:
      - AssetAssignmentReceipt
      - AssetAssignment
      - ToolMovement
      - WorkOrderAudit

    Updates:
      - ToolAsset current holder and status
      - WorkOrderPart assignment state

    This function intentionally does not commit or roll back.
    The calling route owns the transaction.
    """
    part_number = (
        getattr(line, "part_number", "") or ""
    ).strip().upper()

    asset_name = (
        getattr(line, "part_name", "") or part_number
    ).strip()

    requested_quantity = int(
        getattr(line, "quantity", 0) or 0
    )

    holder_id = getattr(work_order, "technician_id", None)
    holder_name = (
        getattr(work_order, "technician_name", "") or ""
    ).strip()

    actor_id = getattr(current_user, "id", None)
    actor_name = (
        getattr(current_user, "username", "") or "system"
    ).strip()

    if not part_number:
        raise AssetAssignmentError(
            "Asset number is missing."
        )

    if not holder_id or not holder_name:
        raise AssetAssignmentError(
            "Select a valid employee before assigning an asset."
        )

    if requested_quantity != 1:
        raise AssetAssignmentError(
            "Asset rows must have quantity 1. "
            "Each reusable asset must be a separate Work Order row."
        )

    if (
        int(getattr(line, "issued_qty", 0) or 0) >= 1
        and getattr(line, "tool_asset_id", None)
    ):
        raise AssetAssignmentError(
            f"Asset {part_number} is already assigned from this Work Order row."
        )

    now = datetime.utcnow()

    inventory_part = _find_inventory_part(
        part_number=part_number,
        warehouse=getattr(line, "warehouse", None),
    )

    inventory_quantity = 0
    if inventory_part is not None:
        try:
            inventory_quantity = max(
                int(inventory_part.quantity or 0),
                0,
            )
        except (TypeError, ValueError):
            inventory_quantity = 0

    asset = (
        ToolAsset.query
        .filter(
            func.upper(ToolAsset.tool_number)
            == part_number
        )
        .first()
    )

    if asset is None:
        asset = ToolAsset(
            tool_number=part_number,
            name=asset_name or part_number,
            asset_type="TOOL",
            quantity=max(inventory_quantity, 1),
            serial_number=None,
            status="available",
            condition="good",
            location="TOOLS",
        )

        db.session.add(asset)
        db.session.flush()

    else:
        if not asset.name:
            asset.name = asset_name or part_number

        if not asset.asset_type:
            asset.asset_type = "TOOL"

        if inventory_quantity > int(asset.quantity or 0):
            asset.quantity = inventory_quantity

        db.session.add(asset)

    free_quantity = _calculate_free_quantity(asset)

    if free_quantity <= 0:
        raise AssetAssignmentError(
            f"Asset {part_number} has no free quantity available."
        )

    old_status = (
        getattr(asset, "status", "") or "available"
    ).strip().lower()

    receipt = AssetAssignmentReceipt(
        receipt_number=_next_receipt_number(),
        assigned_to_id=holder_id,
        assigned_to_name=holder_name,
        issued_by_id=actor_id,
        issued_by_name=actor_name,
        work_order_id=work_order.id,
        assigned_at=now,
        status="open",
        note=f"Created from Work Order #{work_order.id}",
        created_at=now,
        updated_at=now,
    )

    db.session.add(receipt)
    db.session.flush()

    assignment = AssetAssignment(
        receipt_id=receipt.id,
        asset_id=asset.id,
        assigned_to_id=holder_id,
        assigned_to_name=holder_name,
        assigned_by_id=actor_id,
        assigned_by_name=actor_name,
        work_order_id=work_order.id,
        work_unit_id=getattr(line, "unit_id", None),
        quantity=1,
        assigned_at=now,
        status="assigned",
        assignment_note=(
            f"Assigned from Work Order #{work_order.id}"
        ),
        created_at=now,
        updated_at=now,
    )

    db.session.add(assignment)
    db.session.flush()

    movement = ToolMovement(
        tool_id=asset.id,
        work_order_id=work_order.id,
        assignment_id=assignment.id,
        receipt_id=receipt.id,

        # Legacy movement fields remain synchronized.
        technician_id=holder_id,
        technician_name=holder_name,

        action="assigned",
        quantity=1,
        from_status=old_status,
        to_status="assigned",
        note=(
            f"Assigned from Work Order #{work_order.id}; "
            f"receipt {receipt.receipt_number}"
        ),
        actor_user_id=actor_id,
        actor_username=actor_name,
        created_at=now,
    )

    db.session.add(movement)

    # Universal current custody fields.
    asset.status = "assigned"
    asset.location = "EMPLOYEE"
    asset.current_holder_id = holder_id
    asset.current_holder_name = holder_name
    asset.current_work_order_id = work_order.id
    asset.updated_at = now

    # Legacy compatibility.
    asset.current_technician_id = holder_id
    asset.current_technician_name = holder_name

    db.session.add(asset)

    line.tool_asset_id = asset.id
    line.item_type = "tool"
    line.issued_qty = 1
    line.last_issued_at = now
    line.status = "done"
    line.line_status = "done"

    line.backorder_flag = False
    line.ordered_flag = False
    line.ordered_date = None

    db.session.add(line)

    audit = WorkOrderAudit(
        work_order_id=work_order.id,
        action="asset_assigned",
        message=(
            f"Assigned asset {part_number} — "
            f"{asset_name}; receipt {receipt.receipt_number}"
        )[:255],
        meta_json=json.dumps(
            {
                "asset_id": asset.id,
                "assignment_id": assignment.id,
                "receipt_id": receipt.id,
                "receipt_number": receipt.receipt_number,
                "asset_number": part_number,
                "assigned_to": holder_name,
                "audit_reason": "asset_assigned",
            },
            ensure_ascii=False,
        ),
        actor_user_id=actor_id,
        actor_username=actor_name,
        created_at=now,
    )

    db.session.add(audit)

    return AssetAssignmentResult(
        asset=asset,
        assignment=assignment,
        receipt=receipt,
        movement=movement,
    )