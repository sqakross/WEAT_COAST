from app import app
from extensions import db

from models import (
    ApplianceIssue,
    ApplianceIssueLine,
    ApplianceMovement,
    ApplianceUnit,
)

ISSUE_NUMBERS = [
    "AIS-000001",
    "AIS-000002",
]


with app.app_context():

    print("=" * 70)
    print("CLEANUP TEST APPLIANCE ISSUES")
    print("=" * 70)

    issues = (
        ApplianceIssue.query
        .filter(
            ApplianceIssue.issue_number.in_(
                ISSUE_NUMBERS
            )
        )
        .order_by(
            ApplianceIssue.id.asc()
        )
        .all()
    )

    if not issues:
        print("Nothing found.")
        raise SystemExit(0)

    issue_ids = [
        issue.id
        for issue in issues
    ]

    lines = (
        ApplianceIssueLine.query
        .filter(
            ApplianceIssueLine.issue_id.in_(
                issue_ids
            )
        )
        .all()
    )

    unit_ids = sorted(
        {
            line.appliance_unit_id
            for line in lines
        }
    )

    print()
    print("ISSUES TO REMOVE:")

    for issue in issues:
        print(
            f"  {issue.issue_number} | "
            f"tech={issue.technician_username} | "
            f"wo={issue.work_order_number}"
        )

    print()
    print("APPLIANCES TO RESTORE:")

    units = []

    for unit_id in unit_ids:

        unit = db.session.get(
            ApplianceUnit,
            unit_id,
        )

        if unit is None:
            continue

        units.append(unit)

        print(
            f"  {unit.inventory_number} | "
            f"status={unit.status} | "
            f"wo={unit.current_work_order_number}"
        )

    print()
    answer = input(
        "Type YES to remove these test issues: "
    ).strip().upper()

    if answer != "YES":
        print("CANCELLED")
        raise SystemExit(0)

    try:

        # ----------------------------------------------
        # Restore physical units
        # ----------------------------------------------

        for unit in units:

            unit.status = "available"

            unit.current_work_order_id = None
            unit.current_work_order_number = None

        db.session.flush()

        # ----------------------------------------------
        # Delete immutable test movement history
        # ONLY because these are explicit test records.
        # Production history should never be deleted.
        # ----------------------------------------------

        (
            ApplianceMovement.query
            .filter(
                ApplianceMovement.issue_id.in_(
                    issue_ids
                )
            )
            .delete(
                synchronize_session=False
            )
        )

        # ----------------------------------------------
        # Delete lines explicitly.
        # Then expunge issue relationships before
        # deleting headers to avoid cascade warning.
        # ----------------------------------------------

        (
            ApplianceIssueLine.query
            .filter(
                ApplianceIssueLine.issue_id.in_(
                    issue_ids
                )
            )
            .delete(
                synchronize_session=False
            )
        )

        db.session.flush()

        for issue in issues:
            db.session.delete(issue)

        db.session.commit()

        print()
        print("=" * 70)
        print("CLEANUP COMPLETE")
        print("=" * 70)

        print(
            "Removed issues:",
            ", ".join(ISSUE_NUMBERS),
        )

        print()
        print("RESTORED INVENTORY:")

        for unit_id in unit_ids:

            unit = db.session.get(
                ApplianceUnit,
                unit_id,
            )

            if unit is not None:
                print(
                    f"  {unit.inventory_number} | "
                    f"status={unit.status} | "
                    f"wo={unit.current_work_order_number}"
                )

        print()
        print(
            "Remaining matching issues:",
            ApplianceIssue.query
            .filter(
                ApplianceIssue.issue_number.in_(
                    ISSUE_NUMBERS
                )
            )
            .count()
        )

    except Exception:

        db.session.rollback()
        raise
