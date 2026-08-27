from pathlib import Path
from datetime import datetime
import re

path = Path("models.py")

if not path.exists():
    raise SystemExit("ERROR: models.py not found")

text = path.read_text(encoding="utf-8")

if "class ApplianceIssue(db.Model):" in text:
    raise SystemExit(
        "STOP: ApplianceIssue already exists. Nothing changed."
    )

backup = Path(
    f"models.py.before_appliance_issue_"
    f"{datetime.now():%Y%m%d_%H%M%S}.bak"
)

backup.write_text(
    text,
    encoding="utf-8",
)

# ============================================================
# Find ApplianceUnit and insert immediately AFTER its class.
# We locate the next top-level class dynamically.
# ============================================================

start = text.find("class ApplianceUnit(db.Model):")

if start < 0:
    raise SystemExit(
        "ERROR: class ApplianceUnit(db.Model) not found"
    )

match = re.search(
    r"^class\s+[A-Za-z_][A-Za-z0-9_]*\s*[\(:]",
    text[start + 1:],
    flags=re.MULTILINE,
)

if match:
    insert_at = start + 1 + match.start()
else:
    insert_at = len(text)


models_block = r'''

# ============================================================
# Appliance Issue
# Warehouse issue document / technician issue slip
#
# One ApplianceIssue = one AIS document.
# One document may contain many physical appliances.
#
# IMPORTANT:
#   This is NOT a customer financial invoice.
#   Warehouse users do not work with pricing here.
# ============================================================

class ApplianceIssue(db.Model):
    __tablename__ = "appliance_issue"

    __table_args__ = (
        db.UniqueConstraint(
            "issue_number",
            name="uq_appliance_issue_number",
        ),
        db.Index(
            "ix_appliance_issue_warehouse_status",
            "warehouse_id",
            "status",
        ),
        db.Index(
            "ix_appliance_issue_technician_date",
            "technician_id",
            "issued_at",
        ),
        db.Index(
            "ix_appliance_issue_work_order",
            "work_order_id",
        ),
        {"extend_existing": True},
    )

    id = db.Column(
        db.Integer,
        primary_key=True,
    )

    # Human-readable warehouse document number:
    # AIS-000001, AIS-000002, ...
    issue_number = db.Column(
        db.String(40),
        nullable=False,
        unique=True,
        index=True,
    )

    # Warehouse from which the appliances were issued.
    warehouse_id = db.Column(
        db.Integer,
        db.ForeignKey(
            "warehouse.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    # Technician / employee receiving the appliances.
    technician_id = db.Column(
        db.Integer,
        db.ForeignKey(
            "user.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    # Original W/O selected when the issue document is created.
    #
    # Individual appliances may later move to another W/O.
    # That current per-appliance W/O is stored in
    # ApplianceIssueLine.current_work_order_id.
    work_order_id = db.Column(
        db.Integer,
        db.ForeignKey(
            "work_orders.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    # Document status.
    #
    # issued = normal posted warehouse issue
    # void   = reserved for future controlled void workflow
    status = db.Column(
        db.String(30),
        nullable=False,
        default="issued",
        index=True,
    )

    issued_at = db.Column(
        db.DateTime,
        nullable=False,
        default=datetime.utcnow,
        index=True,
    )

    issued_by_id = db.Column(
        db.Integer,
        db.ForeignKey(
            "user.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    notes = db.Column(
        db.Text,
        nullable=True,
    )

    created_at = db.Column(
        db.DateTime,
        nullable=False,
        default=datetime.utcnow,
    )

    updated_at = db.Column(
        db.DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    warehouse = db.relationship(
        "Warehouse",
        foreign_keys=[warehouse_id],
        lazy="joined",
    )

    technician = db.relationship(
        "User",
        foreign_keys=[technician_id],
        lazy="joined",
    )

    work_order = db.relationship(
        "WorkOrder",
        foreign_keys=[work_order_id],
        lazy="joined",
    )

    issued_by = db.relationship(
        "User",
        foreign_keys=[issued_by_id],
        lazy="joined",
    )

    lines = db.relationship(
        "ApplianceIssueLine",
        back_populates="issue",
        cascade="all, delete-orphan",
        lazy="selectin",
        order_by="ApplianceIssueLine.line_no",
    )

    @property
    def issued_at_local(self):
        return utc_to_local(self.issued_at)

    @property
    def created_at_local(self):
        return utc_to_local(self.created_at)

    @property
    def updated_at_local(self):
        return utc_to_local(self.updated_at)

    @property
    def appliance_count(self) -> int:
        return len(self.lines or [])

    @property
    def technician_username(self) -> str:
        if self.technician and self.technician.username:
            return self.technician.username

        return ""

    @property
    def work_order_number(self) -> str:
        if self.work_order is None:
            return ""

        return (
            self.work_order.canonical_job
            or self.work_order.job_numbers
            or ""
        )

    def __repr__(self):
        return (
            f"<ApplianceIssue id={self.id} "
            f"number={self.issue_number!r} "
            f"status={self.status!r}>"
        )


# ============================================================
# Appliance Issue Line
#
# One row = one physical appliance on an Issue Slip.
#
# IMPORTANT:
# Snapshot fields intentionally duplicate appliance data.
# If Model/Serial/etc. is corrected later, an old printed
# warehouse document still represents what was issued then.
# ============================================================

class ApplianceIssueLine(db.Model):
    __tablename__ = "appliance_issue_line"

    __table_args__ = (
        db.UniqueConstraint(
            "issue_id",
            "line_no",
            name="uq_appliance_issue_line_number",
        ),
        db.UniqueConstraint(
            "issue_id",
            "appliance_unit_id",
            name="uq_appliance_issue_line_unit",
        ),
        db.Index(
            "ix_appliance_issue_line_unit_status",
            "appliance_unit_id",
            "status",
        ),
        db.Index(
            "ix_appliance_issue_line_work_order_status",
            "current_work_order_id",
            "status",
        ),
        {"extend_existing": True},
    )

    id = db.Column(
        db.Integer,
        primary_key=True,
    )

    issue_id = db.Column(
        db.Integer,
        db.ForeignKey(
            "appliance_issue.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    line_no = db.Column(
        db.Integer,
        nullable=False,
    )

    appliance_unit_id = db.Column(
        db.Integer,
        db.ForeignKey(
            "appliance_unit.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    # Current W/O for THIS physical appliance.
    #
    # Initially equals ApplianceIssue.work_order_id.
    # CHANGE W/O updates this value, while Movement preserves
    # the previous W/O in immutable history.
    current_work_order_id = db.Column(
        db.Integer,
        db.ForeignKey(
            "work_orders.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    # Operational state of this specific issued appliance:
    #
    # issued
    # installed
    # returned
    # replaced
    # customer_exchange
    #
    # This is NOT financial invoice/payment status.
    status = db.Column(
        db.String(40),
        nullable=False,
        default="issued",
        index=True,
    )

    installed_at = db.Column(
        db.DateTime,
        nullable=True,
        index=True,
    )

    resolved_at = db.Column(
        db.DateTime,
        nullable=True,
    )

    notes = db.Column(
        db.Text,
        nullable=True,
    )

    # --------------------------------------------------------
    # Historical snapshot for the warehouse Issue Slip.
    # No prices are stored here.
    # --------------------------------------------------------

    inventory_number_snapshot = db.Column(
        db.String(40),
        nullable=False,
    )

    appliance_type_snapshot = db.Column(
        db.String(120),
        nullable=True,
    )

    brand_snapshot = db.Column(
        db.String(100),
        nullable=True,
    )

    model_number_snapshot = db.Column(
        db.String(120),
        nullable=True,
    )

    serial_number_snapshot = db.Column(
        db.String(160),
        nullable=True,
    )

    size_value_snapshot = db.Column(
        db.Float,
        nullable=True,
    )

    size_unit_snapshot = db.Column(
        db.String(20),
        nullable=True,
    )

    condition_snapshot = db.Column(
        db.String(40),
        nullable=True,
    )

    created_at = db.Column(
        db.DateTime,
        nullable=False,
        default=datetime.utcnow,
    )

    updated_at = db.Column(
        db.DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    issue = db.relationship(
        "ApplianceIssue",
        foreign_keys=[issue_id],
        back_populates="lines",
    )

    appliance_unit = db.relationship(
        "ApplianceUnit",
        foreign_keys=[appliance_unit_id],
        lazy="joined",
    )

    current_work_order = db.relationship(
        "WorkOrder",
        foreign_keys=[current_work_order_id],
        lazy="joined",
    )

    @property
    def installed_at_local(self):
        return utc_to_local(self.installed_at)

    @property
    def created_at_local(self):
        return utc_to_local(self.created_at)

    @property
    def updated_at_local(self):
        return utc_to_local(self.updated_at)

    @property
    def current_work_order_number(self) -> str:
        if self.current_work_order is None:
            return ""

        return (
            self.current_work_order.canonical_job
            or self.current_work_order.job_numbers
            or ""
        )

    def __repr__(self):
        return (
            f"<ApplianceIssueLine id={self.id} "
            f"issue_id={self.issue_id} "
            f"unit_id={self.appliance_unit_id} "
            f"status={self.status!r}>"
        )


# ============================================================
# Appliance Movement
#
# Immutable operational history for one physical appliance.
#
# We DO NOT rewrite old movement rows.
#
# Expected movement types include:
#
#   RECEIVING
#   ISSUE
#   INSTALLED
#   WORK_ORDER_CHANGE
#   RETURN_TO_STOCK
#   REPLACEMENT_OUT
#   REPLACEMENT_IN
#   CUSTOMER_EXCHANGE_OUT
#   CUSTOMER_EXCHANGE_IN
#   TRANSFER
#   VENDOR_RETURN
#   SOLD
#   WRITE_OFF
#   SERIAL_CORRECTION
# ============================================================

class ApplianceMovement(db.Model):
    __tablename__ = "appliance_movement"

    __table_args__ = (
        db.Index(
            "ix_appliance_movement_unit_created",
            "appliance_unit_id",
            "created_at",
        ),
        db.Index(
            "ix_appliance_movement_issue_created",
            "issue_id",
            "created_at",
        ),
        db.Index(
            "ix_appliance_movement_type_created",
            "movement_type",
            "created_at",
        ),
        {"extend_existing": True},
    )

    id = db.Column(
        db.Integer,
        primary_key=True,
    )

    appliance_unit_id = db.Column(
        db.Integer,
        db.ForeignKey(
            "appliance_unit.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    movement_type = db.Column(
        db.String(50),
        nullable=False,
        index=True,
    )

    issue_id = db.Column(
        db.Integer,
        db.ForeignKey(
            "appliance_issue.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    issue_line_id = db.Column(
        db.Integer,
        db.ForeignKey(
            "appliance_issue_line.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    from_warehouse_id = db.Column(
        db.Integer,
        db.ForeignKey(
            "warehouse.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    to_warehouse_id = db.Column(
        db.Integer,
        db.ForeignKey(
            "warehouse.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    from_work_order_id = db.Column(
        db.Integer,
        db.ForeignKey(
            "work_orders.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    to_work_order_id = db.Column(
        db.Integer,
        db.ForeignKey(
            "work_orders.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    technician_id = db.Column(
        db.Integer,
        db.ForeignKey(
            "user.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    # Used for REPLACE / CUSTOMER EXCHANGE:
    # old AP <-> new AP.
    related_appliance_unit_id = db.Column(
        db.Integer,
        db.ForeignKey(
            "appliance_unit.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    reason_code = db.Column(
        db.String(80),
        nullable=True,
        index=True,
    )

    notes = db.Column(
        db.Text,
        nullable=True,
    )

    # Optional structured audit details.
    # Keep as text JSON to remain SQLite-friendly.
    meta_json = db.Column(
        db.Text,
        nullable=True,
    )

    actor_id = db.Column(
        db.Integer,
        db.ForeignKey(
            "user.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    created_at = db.Column(
        db.DateTime,
        nullable=False,
        default=datetime.utcnow,
        index=True,
    )

    appliance_unit = db.relationship(
        "ApplianceUnit",
        foreign_keys=[appliance_unit_id],
        lazy="joined",
    )

    related_appliance_unit = db.relationship(
        "ApplianceUnit",
        foreign_keys=[related_appliance_unit_id],
        lazy="select",
    )

    issue = db.relationship(
        "ApplianceIssue",
        foreign_keys=[issue_id],
        lazy="select",
    )

    issue_line = db.relationship(
        "ApplianceIssueLine",
        foreign_keys=[issue_line_id],
        lazy="select",
    )

    from_warehouse = db.relationship(
        "Warehouse",
        foreign_keys=[from_warehouse_id],
        lazy="select",
    )

    to_warehouse = db.relationship(
        "Warehouse",
        foreign_keys=[to_warehouse_id],
        lazy="select",
    )

    from_work_order = db.relationship(
        "WorkOrder",
        foreign_keys=[from_work_order_id],
        lazy="select",
    )

    to_work_order = db.relationship(
        "WorkOrder",
        foreign_keys=[to_work_order_id],
        lazy="select",
    )

    technician = db.relationship(
        "User",
        foreign_keys=[technician_id],
        lazy="select",
    )

    actor = db.relationship(
        "User",
        foreign_keys=[actor_id],
        lazy="select",
    )

    @property
    def created_at_local(self):
        return utc_to_local(self.created_at)

    def __repr__(self):
        return (
            f"<ApplianceMovement id={self.id} "
            f"unit_id={self.appliance_unit_id} "
            f"type={self.movement_type!r}>"
        )


'''

new_text = (
    text[:insert_at]
    + models_block
    + text[insert_at:]
)

path.write_text(
    new_text,
    encoding="utf-8",
)

print("=" * 70)
print("APPLIANCE ISSUE FOUNDATION")
print("=" * 70)
print("OK: models.py patched")
print("Added:")
print("  ApplianceIssue")
print("  ApplianceIssueLine")
print("  ApplianceMovement")
print()
print("Backup:", backup)
print()
print("NO DATABASE MIGRATION WAS RUN.")
print("=" * 70)
