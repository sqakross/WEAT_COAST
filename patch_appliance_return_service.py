from pathlib import Path
from datetime import datetime

path = Path("services/appliance_issue_service.py")

if not path.exists():
    raise SystemExit(
        "ERROR: services/appliance_issue_service.py not found"
    )

text = path.read_text(
    encoding="utf-8"
)

stamp = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

backup = Path(
    str(path)
    + f".before_return_to_stock_{stamp}.bak"
)

backup.write_text(
    text,
    encoding="utf-8",
)

print("BACKUP:", backup)


# ============================================================
# Constants
# ============================================================

if 'LINE_STATUS_RETURNED = "returned"' not in text:

    needle = '''    LINE_STATUS_ISSUED = "issued"

    MOVEMENT_ISSUE = "ISSUE"
'''

    replacement = '''    LINE_STATUS_ISSUED = "issued"
    LINE_STATUS_RETURNED = "returned"

    MOVEMENT_ISSUE = "ISSUE"
    MOVEMENT_RETURN_TO_STOCK = "RETURN_TO_STOCK"
'''

    if needle not in text:
        raise SystemExit(
            "ERROR: service constants block not found"
        )

    text = text.replace(
        needle,
        replacement,
        1,
    )

    print("OK: RETURN constants added")


# ============================================================
# RETURN method
# ============================================================

if "def return_to_stock(" not in text:

    method = r'''

    # =========================================================
    # Return one issued appliance back to warehouse stock
    # =========================================================

    @staticmethod
    def return_to_stock(
        *,
        actor: User,
        issue_line_id: int,
        notes: str | None = None,
    ) -> ApplianceIssueLine:
        """
        Return ONE physical appliance from an Issue Slip
        back to its warehouse.

        This is an operational RETURN, not DELETE/VOID.

        History is preserved:

            original ISSUE movement remains;
            Issue Slip remains;
            Issue Line becomes RETURNED;
            new RETURN_TO_STOCK movement is created;
            physical ApplianceUnit becomes AVAILABLE.

        The operation is atomic.
        """

        actor = (
            ApplianceIssueService
            ._require_user(actor)
        )

        # -----------------------------------------------------
        # Load Issue Line
        # -----------------------------------------------------

        try:
            line_id = int(issue_line_id)
        except (TypeError, ValueError):
            raise ApplianceIssueError(
                "Invalid Appliance Issue line."
            )

        line = db.session.get(
            ApplianceIssueLine,
            line_id,
        )

        if line is None:
            raise ApplianceIssueError(
                "Appliance Issue line not found."
            )

        issue = db.session.get(
            ApplianceIssue,
            line.issue_id,
        )

        if issue is None:
            raise ApplianceIssueError(
                "Appliance Issue not found."
            )

        # -----------------------------------------------------
        # Permission follows original warehouse
        # -----------------------------------------------------

        warehouse = (
            ApplianceIssueService
            ._get_warehouse(
                issue.warehouse_id
            )
        )

        ApplianceIssueService._require_permission(
            actor=actor,
            warehouse_id=warehouse.id,
        )

        # -----------------------------------------------------
        # Validate line state
        # -----------------------------------------------------

        line_status = (
            line.status
            or ""
        ).strip().lower()

        if (
            line_status
            != ApplianceIssueService.LINE_STATUS_ISSUED
        ):
            raise ApplianceIssueError(
                f"This appliance cannot be returned. "
                f"Current Issue status: "
                f"{line_status.upper() or 'UNKNOWN'}."
            )

        unit = db.session.get(
            ApplianceUnit,
            line.appliance_unit_id,
        )

        if unit is None:
            raise ApplianceIssueError(
                "Physical appliance not found."
            )

        # It must still be physically issued.
        unit_status = (
            unit.status
            or ""
        ).strip().lower()

        if (
            unit_status
            != ApplianceIssueService.STATUS_ISSUED
        ):
            raise ApplianceIssueError(
                f"{unit.inventory_number} cannot be "
                f"returned from this Issue because its "
                f"current inventory status is "
                f"{unit_status.upper() or 'UNKNOWN'}."
            )

        # -----------------------------------------------------
        # Make sure the appliance still belongs to THIS
        # issue line operationally.
        #
        # If CHANGE W/O happens in the future, return is still
        # valid because the line remains the authoritative
        # issued-appliance record.
        # -----------------------------------------------------

        old_work_order_id = (
            line.current_work_order_id
        )

        old_work_order_number = (
            ApplianceIssueService._clean_text(
                line.current_work_order_number,
                upper=True,
                max_length=120,
            )
        )

        clean_notes = (
            ApplianceIssueService._clean_text(
                notes,
                max_length=2000,
            )
        )

        now = datetime.utcnow()

        # -----------------------------------------------------
        # Concurrency-safe physical inventory transition
        #
        # ISSUED -> AVAILABLE
        #
        # If another request changed this appliance between
        # validation and UPDATE, rowcount becomes 0 and the
        # complete operation is rolled back.
        # -----------------------------------------------------

        try:

            updated_count = (
                db.session.query(
                    ApplianceUnit
                )
                .filter(
                    ApplianceUnit.id
                    == unit.id,

                    ApplianceUnit.status
                    == ApplianceIssueService.STATUS_ISSUED,
                )
                .update(
                    {
                        ApplianceUnit.status:
                            ApplianceIssueService
                            .STATUS_AVAILABLE,

                        ApplianceUnit.current_work_order_id:
                            None,

                        ApplianceUnit.current_work_order_number:
                            None,

                        ApplianceUnit.updated_at:
                            now,

                        ApplianceUnit.updated_by_id:
                            actor.id,
                    },
                    synchronize_session=False,
                )
            )

            if updated_count != 1:
                raise ApplianceIssueError(
                    f"{unit.inventory_number} was changed "
                    f"by another user. Nothing was returned. "
                    f"Refresh and try again."
                )

            # -------------------------------------------------
            # Preserve Issue history.
            # -------------------------------------------------

            line.status = (
                ApplianceIssueService
                .LINE_STATUS_RETURNED
            )

            line.resolved_at = now
            line.updated_at = now

            if clean_notes:
                line.notes = clean_notes

            # Keep current_work_order snapshot on the historical
            # Issue Line. It tells us which W/O the appliance
            # was assigned to immediately before RETURN.
            #
            # The physical ApplianceUnit W/O was cleared above.

            # -------------------------------------------------
            # Immutable movement
            # -------------------------------------------------

            movement = ApplianceMovement(
                appliance_unit_id=unit.id,

                movement_type=(
                    ApplianceIssueService
                    .MOVEMENT_RETURN_TO_STOCK
                ),

                issue_id=issue.id,
                issue_line_id=line.id,

                # Appliance physically returns to the
                # original warehouse.
                from_warehouse_id=None,
                to_warehouse_id=warehouse.id,

                from_work_order_id=(
                    old_work_order_id
                ),
                to_work_order_id=None,

                from_work_order_number=(
                    old_work_order_number
                ),
                to_work_order_number=None,

                technician_id=(
                    issue.technician_id
                ),

                related_appliance_unit_id=None,

                reason_code="NOT_INSTALLED_RETURN",

                notes=clean_notes,

                meta_json=json.dumps(
                    {
                        "issue_number":
                            issue.issue_number,

                        "inventory_number":
                            unit.inventory_number,

                        "appliance_type":
                            line.appliance_type_snapshot,

                        "brand":
                            line.brand_snapshot,

                        "model_number":
                            line.model_number_snapshot,

                        "serial_number":
                            line.serial_number_snapshot,

                        "from_work_order_id":
                            old_work_order_id,

                        "from_work_order_number":
                            old_work_order_number,

                        "warehouse_id":
                            warehouse.id,

                        "warehouse_code":
                            warehouse.code,

                        "technician_id":
                            issue.technician_id,

                        "technician_username":
                            issue.technician_username,

                        "reason":
                            "NOT_INSTALLED_RETURN",
                    },
                    ensure_ascii=False,
                ),

                actor_id=actor.id,
                created_at=now,
            )

            db.session.add(
                movement
            )

            issue.updated_at = now

            # -------------------------------------------------
            # ONE COMMIT
            # -------------------------------------------------

            db.session.commit()

            db.session.refresh(
                line
            )

            return line

        except Exception:
            db.session.rollback()
            raise
'''

    text = text.rstrip() + method + "\n"

    print("OK: return_to_stock() added")

else:
    print(
        "SKIP: return_to_stock() already exists"
    )


path.write_text(
    text,
    encoding="utf-8",
)

print()
print("=" * 70)
print("APPLIANCE RETURN TO STOCK SERVICE PATCH COMPLETE")
print("=" * 70)

