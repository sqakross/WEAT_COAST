from pathlib import Path
from datetime import datetime


STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def backup(path: Path):
    dst = Path(str(path) + f".before_manual_wo_{STAMP}.bak")
    dst.write_text(
        path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    print("BACKUP:", dst)


# ============================================================
# MODELS.PY
# ============================================================

path = Path("models.py")
text = path.read_text(encoding="utf-8")
backup(path)


# ------------------------------------------------------------
# ApplianceUnit: current manual W/O number
# ------------------------------------------------------------

needle = '''    current_work_order_id = db.Column(
        db.Integer,
        db.ForeignKey(
            "work_orders.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    notes = db.Column(
'''

replacement = '''    current_work_order_id = db.Column(
        db.Integer,
        db.ForeignKey(
            "work_orders.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    # Warehouse-side W/O reference.
    # Independent from the main WorkOrder module.
    current_work_order_number = db.Column(
        db.String(120),
        nullable=True,
        index=True,
    )

    notes = db.Column(
'''

if needle not in text:
    raise SystemExit(
        "ERROR: ApplianceUnit current_work_order_id block not found"
    )

text = text.replace(
    needle,
    replacement,
    1,
)


# ------------------------------------------------------------
# ApplianceIssue: manual W/O number
# ------------------------------------------------------------

needle = '''    work_order_id = db.Column(
        db.Integer,
        db.ForeignKey(
            "work_orders.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    # Document status.
'''

replacement = '''    work_order_id = db.Column(
        db.Integer,
        db.ForeignKey(
            "work_orders.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    # Warehouse W/O number entered manually by warehouse staff.
    # This is the authoritative W/O reference for Appliance Issue.
    work_order_number = db.Column(
        db.String(120),
        nullable=False,
        index=True,
    )

    # Document status.
'''

if needle not in text:
    raise SystemExit(
        "ERROR: ApplianceIssue work_order_id block not found"
    )

text = text.replace(
    needle,
    replacement,
    1,
)


# Remove the old property named work_order_number because it
# now becomes a real database column.
old_property = '''    @property
    def work_order_number(self) -> str:
        if self.work_order is None:
            return ""

        return (
            self.work_order.canonical_job
            or self.work_order.job_numbers
            or ""
        )

'''

if old_property not in text:
    raise SystemExit(
        "ERROR: old ApplianceIssue.work_order_number property not found"
    )

text = text.replace(
    old_property,
    "",
    1,
)


# ------------------------------------------------------------
# ApplianceIssueLine: current manual W/O
# ------------------------------------------------------------

needle = '''    current_work_order_id = db.Column(
        db.Integer,
        db.ForeignKey(
            "work_orders.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    # Operational state of this specific issued appliance:
'''

replacement = '''    current_work_order_id = db.Column(
        db.Integer,
        db.ForeignKey(
            "work_orders.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    current_work_order_number = db.Column(
        db.String(120),
        nullable=True,
        index=True,
    )

    # Operational state of this specific issued appliance:
'''

if needle not in text:
    raise SystemExit(
        "ERROR: ApplianceIssueLine W/O block not found"
    )

text = text.replace(
    needle,
    replacement,
    1,
)


old_property = '''    @property
    def current_work_order_number(self) -> str:
        if self.current_work_order is None:
            return ""

        return (
            self.current_work_order.canonical_job
            or self.current_work_order.job_numbers
            or ""
        )

'''

if old_property not in text:
    raise SystemExit(
        "ERROR: old IssueLine current_work_order_number property not found"
    )

text = text.replace(
    old_property,
    "",
    1,
)


# ------------------------------------------------------------
# ApplianceMovement: preserve manual W/O history
# ------------------------------------------------------------

needle = '''    to_work_order_id = db.Column(
        db.Integer,
        db.ForeignKey(
            "work_orders.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    technician_id = db.Column(
'''

replacement = '''    to_work_order_id = db.Column(
        db.Integer,
        db.ForeignKey(
            "work_orders.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    from_work_order_number = db.Column(
        db.String(120),
        nullable=True,
        index=True,
    )

    to_work_order_number = db.Column(
        db.String(120),
        nullable=True,
        index=True,
    )

    technician_id = db.Column(
'''

if needle not in text:
    raise SystemExit(
        "ERROR: ApplianceMovement W/O block not found"
    )

text = text.replace(
    needle,
    replacement,
    1,
)


path.write_text(
    text,
    encoding="utf-8",
)

print("OK: models.py")


# ============================================================
# ISSUE SERVICE
# ============================================================

path = Path("services/appliance_issue_service.py")
text = path.read_text(encoding="utf-8")
backup(path)


# Signature
old = '''        technician_id: int,
        work_order_id: int,
        appliance_unit_ids: list[int],
'''

new = '''        technician_id: int,
        work_order_number: str,
        appliance_unit_ids: list[int],
        work_order_id: int | None = None,
'''

if old not in text:
    raise SystemExit(
        "ERROR: create_issue signature not found"
    )

text = text.replace(
    old,
    new,
    1,
)


# Replace mandatory WorkOrder lookup
old = '''        work_order = (
            ApplianceIssueService._get_work_order(
                work_order_id
            )
        )

        # -----------------------------------------------------
        # Normalize selected IDs
'''

new = '''        work_order_number_clean = (
            ApplianceIssueService._clean_text(
                work_order_number,
                upper=True,
                max_length=120,
            )
        )

        if not work_order_number_clean:
            raise ApplianceIssueError(
                "Work Order # is required."
            )

        # Optional link to main WorkOrder module.
        # Appliance warehouse does NOT depend on it.
        work_order = None

        if work_order_id:
            work_order = db.session.get(
                WorkOrder,
                int(work_order_id),
            )

        # -----------------------------------------------------
        # Normalize selected IDs
'''

if old not in text:
    raise SystemExit(
        "ERROR: mandatory WorkOrder lookup block not found"
    )

text = text.replace(
    old,
    new,
    1,
)


# Issue header
old = '''            technician_id=technician.id,
            work_order_id=work_order.id,
            status=(
'''

new = '''            technician_id=technician.id,
            work_order_id=(
                work_order.id
                if work_order is not None
                else None
            ),
            work_order_number=work_order_number_clean,
            status=(
'''

if old not in text:
    raise SystemExit(
        "ERROR: issue header W/O assignment not found"
    )

text = text.replace(
    old,
    new,
    1,
)


# Inventory bulk update
old = '''                        ApplianceUnit.current_work_order_id:
                            work_order.id,

                        ApplianceUnit.updated_at:
'''

new = '''                        ApplianceUnit.current_work_order_id:
                            (
                                work_order.id
                                if work_order is not None
                                else None
                            ),

                        ApplianceUnit.current_work_order_number:
                            work_order_number_clean,

                        ApplianceUnit.updated_at:
'''

if old not in text:
    raise SystemExit(
        "ERROR: ApplianceUnit W/O update not found"
    )

text = text.replace(
    old,
    new,
    1,
)


# Issue line
old = '''                    appliance_unit_id=unit.id,
                    current_work_order_id=work_order.id,
                    status=(
'''

new = '''                    appliance_unit_id=unit.id,
                    current_work_order_id=(
                        work_order.id
                        if work_order is not None
                        else None
                    ),
                    current_work_order_number=(
                        work_order_number_clean
                    ),
                    status=(
'''

if old not in text:
    raise SystemExit(
        "ERROR: IssueLine W/O assignment not found"
    )

text = text.replace(
    old,
    new,
    1,
)


# Movement
old = '''                    from_work_order_id=None,
                    to_work_order_id=work_order.id,

                    technician_id=technician.id,
'''

new = '''                    from_work_order_id=None,
                    to_work_order_id=(
                        work_order.id
                        if work_order is not None
                        else None
                    ),

                    from_work_order_number=None,
                    to_work_order_number=(
                        work_order_number_clean
                    ),

                    technician_id=technician.id,
'''

if old not in text:
    raise SystemExit(
        "ERROR: Movement W/O assignment not found"
    )

text = text.replace(
    old,
    new,
    1,
)


# Meta JSON work order
old = '''                            "work_order_id":
                                work_order.id,

                            "work_order_number":
                                (
                                    work_order.canonical_job
                                    or work_order.job_numbers
                                    or ""
                                ),
'''

new = '''                            "work_order_id":
                                (
                                    work_order.id
                                    if work_order is not None
                                    else None
                                ),

                            "work_order_number":
                                work_order_number_clean,
'''

if old not in text:
    raise SystemExit(
        "ERROR: Movement meta W/O block not found"
    )

text = text.replace(
    old,
    new,
    1,
)


path.write_text(
    text,
    encoding="utf-8",
)

print("OK: appliance_issue_service.py")


# ============================================================
# ROUTES
# ============================================================

path = Path("appliance/routes.py")
text = path.read_text(encoding="utf-8")
backup(path)


old = '''            work_order_id = int(
                request.form.get("work_order_id")
                or 0
            )

            unit_ids = []
'''

new = '''            work_order_number = (
                request.form.get("work_order_number")
                or ""
            ).strip().upper()

            unit_ids = []
'''

if old not in text:
    raise SystemExit(
        "ERROR: route work_order_id input block not found"
    )

text = text.replace(
    old,
    new,
    1,
)


old = '''                technician_id=technician_id,
                work_order_id=work_order_id,
                appliance_unit_ids=unit_ids,
'''

new = '''                technician_id=technician_id,
                work_order_number=work_order_number,
                appliance_unit_ids=unit_ids,
'''

if old not in text:
    raise SystemExit(
        "ERROR: route create_issue W/O args not found"
    )

text = text.replace(
    old,
    new,
    1,
)


path.write_text(
    text,
    encoding="utf-8",
)

print("OK: appliance/routes.py")


print()
print("=" * 70)
print("MANUAL W/O FOUNDATION PATCH COMPLETE")
print("=" * 70)
