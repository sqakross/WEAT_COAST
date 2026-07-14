from datetime import datetime

from extensions import db
from models import (
    AssetAssignment,
    AssetAssignmentReceipt,
    ToolAsset,
    ToolMovement,
)


def next_receipt_number() -> str:
    """
    Returns the next AR-xxxxxx receipt number.
    """
    latest = (
        AssetAssignmentReceipt.query
        .order_by(AssetAssignmentReceipt.id.desc())
        .first()
    )

    next_number = 1

    if latest and latest.receipt_number:
        raw = latest.receipt_number.strip().upper()

        if raw.startswith("AR-"):
            suffix = raw[3:]

            if suffix.isdigit():
                next_number = int(suffix) + 1

    return f"AR-{next_number:06d}"


def run_backfill() -> None:
    """
    Creates AssetAssignmentReceipt and AssetAssignment records for
    currently assigned legacy ToolAsset records.

    Existing ToolMovement rows are linked to the new records.
    No new ToolMovement is created.
    """
    created_receipts = 0
    created_assignments = 0
    linked_movements = 0
    skipped = 0

    assets = (
        ToolAsset.query
        .filter(ToolAsset.status == "assigned")
        .order_by(ToolAsset.id.asc())
        .all()
    )

    try:
        for asset in assets:
            # Do not create a duplicate active assignment.
            existing_assignment = (
                AssetAssignment.query
                .filter(
                    AssetAssignment.asset_id == asset.id,
                    AssetAssignment.status == "assigned",
                    AssetAssignment.returned_at.is_(None),
                )
                .order_by(AssetAssignment.id.desc())
                .first()
            )

            if existing_assignment:
                print(
                    f"SKIP asset #{asset.id} {asset.tool_number}: "
                    f"active assignment #{existing_assignment.id} already exists"
                )
                skipped += 1
                continue

            movement = (
                ToolMovement.query
                .filter(
                    ToolMovement.tool_id == asset.id,
                    ToolMovement.action == "assigned",
                )
                .order_by(
                    ToolMovement.created_at.desc(),
                    ToolMovement.id.desc(),
                )
                .first()
            )

            if movement is None:
                raise RuntimeError(
                    f"Asset #{asset.id} {asset.tool_number} is assigned, "
                    "but no assigned ToolMovement was found."
                )

            holder_id = (
                asset.current_holder_id
                or asset.current_technician_id
                or movement.technician_id
            )

            holder_name = (
                asset.current_holder_name
                or asset.current_technician_name
                or movement.technician_name
                or ""
            ).strip()

            if not holder_id:
                raise RuntimeError(
                    f"Asset #{asset.id} {asset.tool_number} has no holder user ID."
                )

            if not holder_name:
                raise RuntimeError(
                    f"Asset #{asset.id} {asset.tool_number} has no holder name."
                )

            assigned_at = (
                movement.created_at
                or asset.updated_at
                or datetime.utcnow()
            )

            work_order_id = (
                movement.work_order_id
                or asset.current_work_order_id
            )

            actor_id = movement.actor_user_id
            actor_name = (
                movement.actor_username
                or "legacy migration"
            ).strip()

            receipt = AssetAssignmentReceipt(
                receipt_number=next_receipt_number(),

                assigned_to_id=holder_id,
                assigned_to_name=holder_name,

                issued_by_id=actor_id,
                issued_by_name=actor_name,

                work_order_id=work_order_id,
                assigned_at=assigned_at,

                status="open",
                note=(
                    "Historical receipt created from legacy "
                    f"ToolMovement #{movement.id}."
                ),

                created_at=assigned_at,
                updated_at=datetime.utcnow(),
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

                work_order_id=work_order_id,
                work_unit_id=None,

                quantity=int(movement.quantity or 1),
                assigned_at=assigned_at,
                status="assigned",

                assignment_note=(
                    "Historical assignment created from legacy "
                    f"ToolMovement #{movement.id}."
                ),

                created_at=assigned_at,
                updated_at=datetime.utcnow(),
            )

            db.session.add(assignment)
            db.session.flush()

            # Link the existing historical movement.
            movement.receipt_id = receipt.id
            movement.assignment_id = assignment.id
            db.session.add(movement)

            # Synchronize universal custody fields.
            asset.current_holder_id = holder_id
            asset.current_holder_name = holder_name
            asset.current_work_order_id = work_order_id
            asset.status = "assigned"
            asset.updated_at = datetime.utcnow()
            db.session.add(asset)

            created_receipts += 1
            created_assignments += 1
            linked_movements += 1

            print(
                f"CREATED {receipt.receipt_number}: "
                f"{asset.tool_number} -> {holder_name}, "
                f"WO #{work_order_id}"
            )

        db.session.commit()

    except Exception:
        db.session.rollback()
        raise

    print()
    print("Backfill completed.")
    print(f"Receipts created:   {created_receipts}")
    print(f"Assignments created:{created_assignments}")
    print(f"Movements linked:   {linked_movements}")
    print(f"Skipped:            {skipped}")


run_backfill()
