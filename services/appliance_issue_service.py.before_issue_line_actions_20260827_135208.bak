from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import func

from extensions import db
from models import (
    ApplianceIssue,
    ApplianceIssueLine,
    ApplianceMovement,
    ApplianceUnit,
    User,
    Warehouse,
    WorkOrder,
)
from services.access_control_service import AccessControlService


class ApplianceIssueError(Exception):
    """Expected business error during appliance issue."""


class ApplianceIssueAccessDenied(ApplianceIssueError):
    """User does not have required warehouse permission."""


class ApplianceIssueService:
    """
    Warehouse-side appliance issuing.

    One Issue Slip may contain one or many physical appliances.

    IMPORTANT
    ---------
    This service contains NO customer pricing logic.

    Warehouse workflow:
        AVAILABLE
            -> ISSUE
            -> ISSUED

    Financial Customer Invoice is a separate future workflow.
    """

    STATUS_AVAILABLE = "available"
    STATUS_ISSUED = "issued"

    ISSUE_STATUS_ISSUED = "issued"
    LINE_STATUS_ISSUED = "issued"
    LINE_STATUS_RETURNED = "returned"

    MOVEMENT_ISSUE = "ISSUE"
    MOVEMENT_RETURN_TO_STOCK = "RETURN_TO_STOCK"

    # For now use the existing warehouse-operational permission.
    # This avoids breaking the access profiles already configured
    # for the Appliance module.
    ISSUE_PERMISSION = "appliance.receive"

    MAX_UNITS_PER_ISSUE = 100


    # =========================================================
    # Basic validation helpers
    # =========================================================

    @staticmethod
    def _require_user(actor: User | None) -> User:
        if actor is None:
            raise ApplianceIssueAccessDenied(
                "Authenticated user is required."
            )

        if getattr(actor, "id", None) is None:
            raise ApplianceIssueAccessDenied(
                "Authenticated user is required."
            )

        return actor


    @staticmethod
    def _clean_text(
        value,
        *,
        upper: bool = False,
        max_length: int | None = None,
    ) -> str | None:
        if value is None:
            return None

        value = str(value).strip()

        if not value:
            return None

        if upper:
            value = value.upper()

        if max_length is not None:
            value = value[:max_length]

        return value


    @staticmethod
    def _require_permission(
        *,
        actor: User,
        warehouse_id: int,
    ) -> None:
        allowed = AccessControlService.can(
            actor,
            ApplianceIssueService.ISSUE_PERMISSION,
            warehouse_id=warehouse_id,
        )

        if not allowed:
            raise ApplianceIssueAccessDenied(
                "You do not have permission to issue "
                "appliances from this warehouse."
            )


    @staticmethod
    def _get_warehouse(
        warehouse_id: int,
    ) -> Warehouse:
        warehouse = db.session.get(
            Warehouse,
            int(warehouse_id),
        )

        if warehouse is None:
            raise ApplianceIssueError(
                "Warehouse not found."
            )

        if not warehouse.is_active:
            raise ApplianceIssueError(
                "Warehouse is inactive."
            )

        return warehouse


    @staticmethod
    def _get_technician(
        technician_id: int,
    ) -> User:
        technician = db.session.get(
            User,
            int(technician_id),
        )

        if technician is None:
            raise ApplianceIssueError(
                "Technician not found."
            )

        role = (
            getattr(technician, "role", "")
            or ""
        ).strip().lower()

        if role != "technician":
            raise ApplianceIssueError(
                f"{technician.username} is not a technician."
            )

        return technician


    @staticmethod
    def _get_work_order(
        work_order_id: int,
    ) -> WorkOrder:
        work_order = db.session.get(
            WorkOrder,
            int(work_order_id),
        )

        if work_order is None:
            raise ApplianceIssueError(
                "Work Order not found."
            )

        return work_order


    # =========================================================
    # Issue numbering
    # =========================================================

    @staticmethod
    def _next_issue_number() -> str:
        """
        Human-readable warehouse Issue Slip number.

        Example:
            AIS-000001
            AIS-000002

        The database UNIQUE constraint remains the final
        protection against accidental duplicate numbers.
        """

        max_id = (
            db.session.query(
                func.coalesce(
                    func.max(ApplianceIssue.id),
                    0,
                )
            )
            .scalar()
            or 0
        )

        return f"AIS-{int(max_id) + 1:06d}"


    # =========================================================
    # Public read helpers
    # =========================================================

    @staticmethod
    def available_units(
        *,
        actor: User,
        warehouse_id: int,
    ) -> list[ApplianceUnit]:
        actor = ApplianceIssueService._require_user(
            actor
        )

        warehouse = (
            ApplianceIssueService._get_warehouse(
                warehouse_id
            )
        )

        ApplianceIssueService._require_permission(
            actor=actor,
            warehouse_id=warehouse.id,
        )

        return (
            ApplianceUnit.query
            .filter(
                ApplianceUnit.warehouse_id
                == warehouse.id,
                ApplianceUnit.status
                == ApplianceIssueService.STATUS_AVAILABLE,
            )
            .order_by(
                ApplianceUnit.category_id.asc(),
                ApplianceUnit.inventory_number.asc(),
            )
            .all()
        )


    @staticmethod
    def get_issue(
        *,
        issue_id: int,
        actor: User,
    ) -> ApplianceIssue:
        actor = ApplianceIssueService._require_user(
            actor
        )

        issue = db.session.get(
            ApplianceIssue,
            int(issue_id),
        )

        if issue is None:
            raise ApplianceIssueError(
                "Appliance Issue not found."
            )

        ApplianceIssueService._require_permission(
            actor=actor,
            warehouse_id=issue.warehouse_id,
        )

        return issue


    # =========================================================
    # Create Issue Slip
    # =========================================================

    @staticmethod
    def create_issue(
        *,
        actor: User,
        warehouse_id: int,
        technician_id: int,
        work_order_number: str,
        appliance_unit_ids: list[int],
        work_order_id: int | None = None,
        notes: str | None = None,
    ) -> ApplianceIssue:
        """
        Issue one or many physical appliances atomically.

        ALL selected units must:
            - exist;
            - be AVAILABLE;
            - belong to the selected warehouse.

        If even one unit fails validation:
            -> nothing is issued
            -> no Issue Slip is created
            -> no Movement is created

        On success:
            ApplianceIssue
            ApplianceIssueLine x N
            ApplianceMovement x N
            ApplianceUnit.status -> issued
            ApplianceUnit.current_work_order_id -> W/O
        """

        actor = ApplianceIssueService._require_user(
            actor
        )

        # -----------------------------------------------------
        # Header validation
        # -----------------------------------------------------

        warehouse = (
            ApplianceIssueService._get_warehouse(
                warehouse_id
            )
        )

        ApplianceIssueService._require_permission(
            actor=actor,
            warehouse_id=warehouse.id,
        )

        technician = (
            ApplianceIssueService._get_technician(
                technician_id
            )
        )

        work_order_number_clean = (
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
        # -----------------------------------------------------

        if not isinstance(
            appliance_unit_ids,
            (list, tuple, set),
        ):
            raise ApplianceIssueError(
                "Appliance selection must be a list."
            )

        normalized_ids = []

        seen_ids = set()

        for raw_id in appliance_unit_ids:
            try:
                unit_id = int(raw_id)
            except (TypeError, ValueError):
                raise ApplianceIssueError(
                    f"Invalid appliance ID: {raw_id}"
                )

            if unit_id <= 0:
                raise ApplianceIssueError(
                    f"Invalid appliance ID: {raw_id}"
                )

            if unit_id in seen_ids:
                continue

            seen_ids.add(unit_id)
            normalized_ids.append(unit_id)

        if not normalized_ids:
            raise ApplianceIssueError(
                "Select at least one appliance."
            )

        if (
            len(normalized_ids)
            > ApplianceIssueService.MAX_UNITS_PER_ISSUE
        ):
            raise ApplianceIssueError(
                "Maximum 100 appliances per Issue Slip."
            )

        # -----------------------------------------------------
        # Load all selected physical units in one query.
        # -----------------------------------------------------

        units = (
            ApplianceUnit.query
            .filter(
                ApplianceUnit.id.in_(
                    normalized_ids
                )
            )
            .order_by(
                ApplianceUnit.id.asc()
            )
            .all()
        )

        units_by_id = {
            int(unit.id): unit
            for unit in units
        }

        # Make sure every requested unit actually exists.
        missing_ids = [
            unit_id
            for unit_id in normalized_ids
            if unit_id not in units_by_id
        ]

        if missing_ids:
            raise ApplianceIssueError(
                "One or more selected appliances "
                "no longer exist."
            )

        # Preserve the user's selected order.
        ordered_units = [
            units_by_id[unit_id]
            for unit_id in normalized_ids
        ]

        # -----------------------------------------------------
        # Validate ALL units before writing anything.
        # -----------------------------------------------------

        for unit in ordered_units:

            if int(unit.warehouse_id) != int(
                warehouse.id
            ):
                raise ApplianceIssueError(
                    f"{unit.inventory_number} belongs "
                    f"to another warehouse."
                )

            status = (
                unit.status
                or ""
            ).strip().lower()

            if (
                status
                != ApplianceIssueService.STATUS_AVAILABLE
            ):
                raise ApplianceIssueError(
                    f"{unit.inventory_number} is not "
                    f"available. Current status: "
                    f"{status or 'UNKNOWN'}."
                )

        now = datetime.utcnow()

        issue_number = (
            ApplianceIssueService._next_issue_number()
        )

        issue = ApplianceIssue(
            issue_number=issue_number,
            warehouse_id=warehouse.id,
            technician_id=technician.id,
            work_order_id=(
                work_order.id
                if work_order is not None
                else None
            ),
            work_order_number=work_order_number_clean,
            status=(
                ApplianceIssueService
                .ISSUE_STATUS_ISSUED
            ),
            issued_at=now,
            issued_by_id=actor.id,
            notes=(
                ApplianceIssueService._clean_text(
                    notes
                )
            ),
            created_at=now,
            updated_at=now,
        )

        # -----------------------------------------------------
        # Add header and FLUSH only.
        #
        # No COMMIT yet. We still need all lines + movements +
        # current inventory state to succeed together.
        # -----------------------------------------------------

        db.session.add(issue)

        try:
            db.session.flush()

            # -------------------------------------------------
            # Concurrency guard
            #
            # Conditional UPDATE means that if another request
            # issued one of these appliances after our SELECT,
            # rowcount will be less than expected and the whole
            # transaction is rolled back.
            # -------------------------------------------------

            updated_count = (
                db.session.query(
                    ApplianceUnit
                )
                .filter(
                    ApplianceUnit.id.in_(
                        normalized_ids
                    ),
                    ApplianceUnit.warehouse_id
                    == warehouse.id,
                    ApplianceUnit.status
                    == (
                        ApplianceIssueService
                        .STATUS_AVAILABLE
                    ),
                )
                .update(
                    {
                        ApplianceUnit.status:
                            ApplianceIssueService
                            .STATUS_ISSUED,

                        ApplianceUnit.current_work_order_id:
                            (
                                work_order.id
                                if work_order is not None
                                else None
                            ),

                        ApplianceUnit.current_work_order_number:
                            work_order_number_clean,

                        ApplianceUnit.updated_at:
                            now,

                        ApplianceUnit.updated_by_id:
                            actor.id,
                    },
                    synchronize_session=False,
                )
            )

            if updated_count != len(
                normalized_ids
            ):
                raise ApplianceIssueError(
                    "One or more appliances were "
                    "changed by another user. "
                    "Nothing was issued. Refresh and try again."
                )

            # -------------------------------------------------
            # Create Issue Lines and immutable Movements.
            # -------------------------------------------------

            for line_no, unit in enumerate(
                ordered_units,
                start=1,
            ):

                category_name = (
                    unit.category.name
                    if unit.category is not None
                    else None
                )

                line = ApplianceIssueLine(
                    issue_id=issue.id,
                    line_no=line_no,
                    appliance_unit_id=unit.id,
                    current_work_order_id=(
                        work_order.id
                        if work_order is not None
                        else None
                    ),
                    current_work_order_number=(
                        work_order_number_clean
                    ),
                    status=(
                        ApplianceIssueService
                        .LINE_STATUS_ISSUED
                    ),
                    installed_at=None,
                    resolved_at=None,
                    notes=None,

                    # Historical snapshot for printing.
                    # NO PRICES.
                    inventory_number_snapshot=(
                        unit.inventory_number
                    ),
                    appliance_type_snapshot=(
                        category_name
                    ),
                    brand_snapshot=unit.brand,
                    model_number_snapshot=(
                        unit.model_number
                    ),
                    serial_number_snapshot=(
                        unit.serial_number
                    ),
                    size_value_snapshot=(
                        unit.size_value
                    ),
                    size_unit_snapshot=(
                        unit.size_unit
                    ),
                    condition_snapshot=(
                        unit.condition
                    ),

                    created_at=now,
                    updated_at=now,
                )

                db.session.add(line)
                db.session.flush()

                movement = ApplianceMovement(
                    appliance_unit_id=unit.id,
                    movement_type=(
                        ApplianceIssueService
                        .MOVEMENT_ISSUE
                    ),
                    issue_id=issue.id,
                    issue_line_id=line.id,

                    # Physically leaving AVAILABLE warehouse
                    # inventory for technician installation.
                    from_warehouse_id=warehouse.id,
                    to_warehouse_id=None,

                    from_work_order_id=None,
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
                    related_appliance_unit_id=None,

                    reason_code="TECHNICIAN_ISSUE",

                    notes=None,

                    meta_json=json.dumps(
                        {
                            "issue_number":
                                issue.issue_number,

                            "warehouse_code":
                                warehouse.code,

                            "technician_username":
                                technician.username,

                            "work_order_id":
                                (
                                    work_order.id
                                    if work_order is not None
                                    else None
                                ),

                            "work_order_number":
                                work_order_number_clean,

                            "inventory_number":
                                unit.inventory_number,

                            "appliance_type":
                                category_name,

                            "brand":
                                unit.brand,

                            "model_number":
                                unit.model_number,

                            "serial_number":
                                unit.serial_number,
                        },
                        ensure_ascii=False,
                    ),

                    actor_id=actor.id,
                    created_at=now,
                )

                db.session.add(movement)

            # -------------------------------------------------
            # ONE COMMIT FOR THE ENTIRE ISSUE.
            # -------------------------------------------------

            db.session.commit()

            db.session.refresh(issue)

            return issue

        except Exception:
            db.session.rollback()
            raise

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

