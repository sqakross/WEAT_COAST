from __future__ import annotations

from app import app
from extensions import db

from models import (
    ApplianceIssue,
    ApplianceIssueLine,
    ApplianceMovement,
    ApplianceUnit,
    User,
    WorkOrder,
)

from services.appliance_issue_service import (
    ApplianceIssueService,
)


INVENTORY_NUMBERS = [
    "AP-0000001",
    "AP-0000002",
]

TECHNICIAN_ID = 20      # ART
WORK_ORDER_ID = 5491    # WO 1021162


def main():
    with app.app_context():

        print("=" * 70)
        print("APPLIANCE ISSUE FLOW TEST")
        print("=" * 70)

        actor = (
            User.query
            .filter_by(role="superadmin")
            .order_by(User.id.asc())
            .first()
        )

        if actor is None:
            raise RuntimeError(
                "No superadmin user found."
            )

        technician = db.session.get(
            User,
            TECHNICIAN_ID,
        )

        if technician is None:
            raise RuntimeError(
                f"Technician ID {TECHNICIAN_ID} not found."
            )

        work_order = db.session.get(
            WorkOrder,
            WORK_ORDER_ID,
        )

        if work_order is None:
            raise RuntimeError(
                f"Work Order ID {WORK_ORDER_ID} not found."
            )

        print()
        print("ACTOR:")
        print(
            f"  {actor.id} | "
            f"{actor.username} | "
            f"{actor.role}"
        )

        print()
        print("TECHNICIAN:")
        print(
            f"  {technician.id} | "
            f"{technician.username}"
        )

        print()
        print("WORK ORDER:")
        print(
            f"  ID={work_order.id} | "
            f"WO={work_order.job_numbers} | "
            f"TECH={work_order.technician_name}"
        )

        units = (
            ApplianceUnit.query
            .filter(
                ApplianceUnit.inventory_number.in_(
                    INVENTORY_NUMBERS
                )
            )
            .order_by(
                ApplianceUnit.inventory_number.asc()
            )
            .all()
        )

        if len(units) != len(INVENTORY_NUMBERS):
            found = {
                unit.inventory_number
                for unit in units
            }

            missing = [
                number
                for number in INVENTORY_NUMBERS
                if number not in found
            ]

            raise RuntimeError(
                f"Missing ApplianceUnit(s): {missing}"
            )

        print()
        print("BEFORE ISSUE:")

        for unit in units:
            print(
                f"  {unit.inventory_number} | "
                f"status={unit.status} | "
                f"warehouse_id={unit.warehouse_id} | "
                f"wo={unit.current_work_order_id}"
            )

        # ----------------------------------------------------
        # Safety: do not touch units unless both are AVAILABLE.
        # ----------------------------------------------------

        unavailable = [
            unit.inventory_number
            for unit in units
            if (unit.status or "").strip().lower()
            != "available"
        ]

        if unavailable:
            raise RuntimeError(
                "TEST ABORTED. "
                "These appliances are not AVAILABLE: "
                + ", ".join(unavailable)
            )

        warehouse_ids = {
            int(unit.warehouse_id)
            for unit in units
        }

        if len(warehouse_ids) != 1:
            raise RuntimeError(
                "TEST ABORTED. Appliances are in "
                "different warehouses."
            )

        warehouse_id = next(
            iter(warehouse_ids)
        )

        # ----------------------------------------------------
        # Preserve exact original state for cleanup.
        # ----------------------------------------------------

        original_state = {
            unit.id: {
                "status": unit.status,
                "current_work_order_id":
                    unit.current_work_order_id,
                "updated_at":
                    unit.updated_at,
                "updated_by_id":
                    unit.updated_by_id,
            }
            for unit in units
        }

        issue = None

        try:
            print()
            print("CREATING ISSUE...")

            issue = (
                ApplianceIssueService.create_issue(
                    actor=actor,
                    warehouse_id=warehouse_id,
                    technician_id=technician.id,
                    work_order_id=work_order.id,
                    appliance_unit_ids=[
                        unit.id
                        for unit in units
                    ],
                    notes="AUTOMATED ISSUE FLOW TEST",
                )
            )

            print()
            print("1. ISSUE CREATED")
            print(
                f"   ID: {issue.id}"
            )
            print(
                f"   Number: {issue.issue_number}"
            )
            print(
                f"   Status: {issue.status}"
            )
            print(
                f"   Technician: "
                f"{issue.technician.username}"
            )
            print(
                f"   W/O: "
                f"{issue.work_order.job_numbers}"
            )

            lines = (
                ApplianceIssueLine.query
                .filter_by(
                    issue_id=issue.id
                )
                .order_by(
                    ApplianceIssueLine.line_no.asc()
                )
                .all()
            )

            print()
            print("2. ISSUE LINES")
            print(
                f"   Count: {len(lines)}"
            )

            for line in lines:
                print(
                    f"   {line.line_no}. "
                    f"{line.inventory_number_snapshot} | "
                    f"{line.appliance_type_snapshot} | "
                    f"{line.brand_snapshot} | "
                    f"{line.model_number_snapshot} | "
                    f"{line.serial_number_snapshot} | "
                    f"status={line.status}"
                )

            if len(lines) != 2:
                raise RuntimeError(
                    "Expected exactly 2 Issue Lines."
                )

            movements = (
                ApplianceMovement.query
                .filter_by(
                    issue_id=issue.id
                )
                .order_by(
                    ApplianceMovement.id.asc()
                )
                .all()
            )

            print()
            print("3. MOVEMENTS")
            print(
                f"   Count: {len(movements)}"
            )

            for movement in movements:
                print(
                    f"   {movement.id} | "
                    f"{movement.movement_type} | "
                    f"unit={movement.appliance_unit.inventory_number} | "
                    f"from_wh={movement.from_warehouse_id} | "
                    f"to_wo={movement.to_work_order_id} | "
                    f"tech={movement.technician_id}"
                )

            if len(movements) != 2:
                raise RuntimeError(
                    "Expected exactly 2 ISSUE Movements."
                )

            for movement in movements:
                if movement.movement_type != "ISSUE":
                    raise RuntimeError(
                        "Unexpected movement type: "
                        f"{movement.movement_type}"
                    )

            db.session.expire_all()

            issued_units = (
                ApplianceUnit.query
                .filter(
                    ApplianceUnit.inventory_number.in_(
                        INVENTORY_NUMBERS
                    )
                )
                .order_by(
                    ApplianceUnit.inventory_number.asc()
                )
                .all()
            )

            print()
            print("4. INVENTORY AFTER ISSUE")

            for unit in issued_units:
                print(
                    f"   {unit.inventory_number} | "
                    f"status={unit.status} | "
                    f"wo={unit.current_work_order_id}"
                )

                if unit.status != "issued":
                    raise RuntimeError(
                        f"{unit.inventory_number} "
                        "was not changed to ISSUED."
                    )

                if (
                    unit.current_work_order_id
                    != work_order.id
                ):
                    raise RuntimeError(
                        f"{unit.inventory_number} "
                        "has incorrect current W/O."
                    )

            print()
            print("=" * 70)
            print(
                "APPLIANCE ISSUE FLOW TEST: PASS"
            )
            print("=" * 70)

        finally:
            # =================================================
            # CLEANUP
            # =================================================

            db.session.rollback()

            if issue is not None:

                issue_id = issue.id

                print()
                print("CLEANUP...")

                # Movement must be removed before Issue Lines.
                (
                    ApplianceMovement.query
                    .filter(
                        ApplianceMovement.issue_id
                        == issue_id
                    )
                    .delete(
                        synchronize_session=False
                    )
                )

                (
                    ApplianceIssueLine.query
                    .filter(
                        ApplianceIssueLine.issue_id
                        == issue_id
                    )
                    .delete(
                        synchronize_session=False
                    )
                )

                db.session.delete(
                    db.session.get(
                        ApplianceIssue,
                        issue_id,
                    )
                )

                # Restore exact original ApplianceUnit state.
                for unit_id, old in original_state.items():
                    unit = db.session.get(
                        ApplianceUnit,
                        unit_id,
                    )

                    unit.status = old["status"]
                    unit.current_work_order_id = (
                        old["current_work_order_id"]
                    )
                    unit.updated_at = old["updated_at"]
                    unit.updated_by_id = (
                        old["updated_by_id"]
                    )

                db.session.commit()

                remaining_issue = (
                    db.session.get(
                        ApplianceIssue,
                        issue_id,
                    )
                )

                restored_units = (
                    ApplianceUnit.query
                    .filter(
                        ApplianceUnit.inventory_number.in_(
                            INVENTORY_NUMBERS
                        )
                    )
                    .order_by(
                        ApplianceUnit.inventory_number.asc()
                    )
                    .all()
                )

                print(
                    "   Test issue removed:",
                    remaining_issue is None,
                )

                print(
                    "   Appliance states restored:"
                )

                for unit in restored_units:
                    print(
                        f"     {unit.inventory_number} | "
                        f"status={unit.status} | "
                        f"wo={unit.current_work_order_id}"
                    )

                print()
                print("CLEANUP OK")


if __name__ == "__main__":
    main()
