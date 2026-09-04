from __future__ import annotations
from license_client.operation_gate import authorize_mutation as _authorize_mutation

import json
from datetime import datetime

from sqlalchemy import func

from extensions import db
from models import (
    ApplianceIssue,
    ApplianceIssueLine,
    ApplianceMovement,
    ApplianceRepairOrder,
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
    LINE_STATUS_REPLACED = "replaced"

    MOVEMENT_ISSUE = "ISSUE"
    MOVEMENT_RETURN_TO_STOCK = "RETURN_TO_STOCK"
    MOVEMENT_REPLACE_OUT = "REPLACE_OUT"
    MOVEMENT_REPLACE_IN = "REPLACE_IN"

    STATUS_REPAIR = "repair"
    STATUS_VENDOR_RETURN_PENDING = "vendor_return_pending"
    STATUS_VENDOR_RETURN = "vendor_return"
    STATUS_WRITTEN_OFF = "written_off"

    MOVEMENT_SEND_TO_REPAIR = "SEND_TO_REPAIR"
    MOVEMENT_RETURN_FROM_REPAIR = "RETURN_FROM_REPAIR"
    MOVEMENT_VENDOR_RETURN_PENDING = "VENDOR_RETURN_PENDING"
    MOVEMENT_VENDOR_RETURN = "VENDOR_RETURN"
    MOVEMENT_VENDOR_RETURN_CANCEL = "VENDOR_RETURN_CANCEL"
    MOVEMENT_SCRAP = "SCRAP"

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


    @staticmethod
    def _next_repair_number() -> str:
        """
        Human-readable internal WCCR Repair Order number.

        Example:
            REP-000001
            REP-000002

        The database UNIQUE constraint remains the final
        protection against accidental duplicate numbers.

        The current application uses the same single-writer
        numbering approach already established for Issue
        documents.
        """

        max_id = (
            db.session.query(
                func.coalesce(
                    func.max(ApplianceRepairOrder.id),
                    0,
                )
            )
            .scalar()
            or 0
        )

        return f"REP-{int(max_id) + 1:06d}"


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

        _authorize_mutation("appliance.issue.create")

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
    # Delete incorrectly-created Appliance Issue
    # =========================================================

    @staticmethod
    def delete_issue_correction(
        *,
        actor: User,
        issue_id: int,
    ) -> dict:
        """
        SUPERADMIN correction tool.

        This is NOT a normal warehouse RETURN.

        It physically removes an incorrectly-created Appliance
        Issue only while every appliance is still untouched
        after that Issue's original ISSUE movement.

        Existing behavior is intentionally preserved from the
        former route-level implementation.

        One transaction.
        """
        actor = (
            ApplianceIssueService
            ._require_user(actor)
        )

        # -----------------------------------------------------
        # SUPERADMIN only
        # -----------------------------------------------------

        role = (
            getattr(
                actor,
                "role",
                "",
            )
            or ""
        ).strip().lower()

        if role != "superadmin":
            raise ApplianceIssueAccessDenied(
                "Only SUPERADMIN can delete an Appliance Issue."
            )

        # -----------------------------------------------------
        # Validate Issue ID / load Issue
        # -----------------------------------------------------

        try:
            issue_id_int = int(issue_id)
        except (
            TypeError,
            ValueError,
        ):
            raise ApplianceIssueError(
                "Invalid Appliance Issue."
            )

        issue = db.session.get(
            ApplianceIssue,
            issue_id_int,
        )

        if issue is None:
            raise ApplianceIssueError(
                "Appliance Issue not found."
            )

        # -----------------------------------------------------
        # Preserve existing warehouse-access rule.
        # -----------------------------------------------------

        if not AccessControlService.can(
            actor,
            "appliance.receive",
            warehouse_id=issue.warehouse_id,
        ):
            raise ApplianceIssueAccessDenied(
                "You do not have access to this warehouse."
            )

        # -----------------------------------------------------
        # Lines / physical units
        # -----------------------------------------------------

        lines = (
            ApplianceIssueLine.query
            .filter(
                ApplianceIssueLine.issue_id
                == issue.id
            )
            .order_by(
                ApplianceIssueLine.line_no.asc()
            )
            .all()
        )

        if not lines:
            raise ApplianceIssueError(
                "Issue contains no appliance lines."
            )

        unit_ids = [
            int(line.appliance_unit_id)
            for line in lines
        ]

        units = (
            ApplianceUnit.query
            .filter(
                ApplianceUnit.id.in_(
                    unit_ids
                )
            )
            .all()
        )

        units_by_id = {
            int(unit.id): unit
            for unit in units
        }

        if len(units_by_id) != len(
            set(unit_ids)
        ):
            raise ApplianceIssueError(
                "One or more appliance units "
                "from this Issue no longer exist."
            )

        # -----------------------------------------------------
        # Find original ISSUE movements belonging to this AIS.
        # -----------------------------------------------------

        original_issue_movements = (
            ApplianceMovement.query
            .filter(
                ApplianceMovement.issue_id
                == issue.id,
                ApplianceMovement.movement_type
                == "ISSUE",
            )
            .all()
        )

        # Kept intentionally for behavioral parity / diagnostics.
        original_movement_ids = [
            movement.id
            for movement
            in original_issue_movements
        ]

        # -----------------------------------------------------
        # SAFETY CHECK
        #
        # Delete is allowed only if THIS Issue is still the
        # latest operational movement for every appliance line.
        #
        # Historical movements before this Issue do not block.
        # -----------------------------------------------------

        issue_movements_by_line = {
            movement.issue_line_id: movement
            for movement in original_issue_movements
            if movement.issue_line_id is not None
        }

        for line in lines:

            issue_movement = (
                issue_movements_by_line.get(
                    line.id
                )
            )

            if issue_movement is None:
                raise ApplianceIssueError(
                    f"Original ISSUE movement is missing "
                    f"for line #{line.line_no}. "
                    "Issue deletion is blocked."
                )

            later_movement = (
                ApplianceMovement.query
                .filter(
                    ApplianceMovement.appliance_unit_id
                    == line.appliance_unit_id,

                    db.or_(
                        ApplianceMovement.created_at
                        > issue_movement.created_at,

                        (
                            ApplianceMovement.created_at
                            == issue_movement.created_at
                        )
                        & (
                            ApplianceMovement.id
                            > issue_movement.id
                        ),
                    ),
                )
                .order_by(
                    ApplianceMovement.created_at.asc(),
                    ApplianceMovement.id.asc(),
                )
                .first()
            )

            if later_movement is not None:

                unit = units_by_id[
                    int(line.appliance_unit_id)
                ]

                raise ApplianceIssueError(
                    f"{unit.inventory_number} has movement "
                    f"{later_movement.movement_type} after "
                    f"{issue.issue_number}. "
                    "Issue deletion is no longer allowed. "
                    "Use RETURN / REPLACE / CHANGE W/O instead."
                )

        # -----------------------------------------------------
        # Current-state validation
        # -----------------------------------------------------

        for line in lines:

            unit = units_by_id[
                int(line.appliance_unit_id)
            ]

            status = (
                unit.status
                or ""
            ).strip().lower()

            line_status = (
                line.status
                or ""
            ).strip().lower()

            # An untouched Issue line is still ISSUED.
            if (
                line_status == "issued"
                and status != "issued"
            ):
                raise ApplianceIssueError(
                    f"{unit.inventory_number} has unexpected "
                    f"inventory status "
                    f"{status.upper() or 'UNKNOWN'}. "
                    "Issue deletion is blocked."
                )

            # Preserve existing route behavior for RETURNED.
            if (
                line_status == "returned"
                and status != "available"
            ):
                raise ApplianceIssueError(
                    f"{unit.inventory_number} was returned "
                    "from this Issue but is no longer AVAILABLE. "
                    "Issue deletion is blocked."
                )

            if line_status not in (
                "issued",
                "returned",
            ):
                raise ApplianceIssueError(
                    f"{unit.inventory_number} has Issue status "
                    f"{line_status.upper() or 'UNKNOWN'}. "
                    "Issue deletion is no longer allowed."
                )

            current_wo = (
                unit.current_work_order_number
                or ""
            ).strip().upper()

            issue_wo = (
                issue.work_order_number
                or ""
            ).strip().upper()

            if (
                current_wo
                and issue_wo
                and current_wo != issue_wo
            ):
                raise ApplianceIssueError(
                    f"{unit.inventory_number} is already "
                    "assigned to another Work Order. "
                    "Issue deletion is blocked."
                )

        issue_number = issue.issue_number
        deleted_unit_count = len(units)

        now = datetime.utcnow()

        try:
            # -------------------------------------------------
            # Restore physical inventory
            # -------------------------------------------------

            for unit in units:

                unit.status = "available"

                unit.current_work_order_id = None
                unit.current_work_order_number = None

                unit.updated_at = now
                unit.updated_by_id = actor.id

            db.session.flush()

            # -------------------------------------------------
            # Remove movements belonging to this Issue
            # -------------------------------------------------

            deleted_movements = (
                ApplianceMovement.query
                .filter(
                    ApplianceMovement.issue_id
                    == issue.id
                )
                .delete(
                    synchronize_session=False
                )
            )

            # -------------------------------------------------
            # Remove Issue Lines
            # -------------------------------------------------

            deleted_lines = (
                ApplianceIssueLine.query
                .filter(
                    ApplianceIssueLine.issue_id
                    == issue.id
                )
                .delete(
                    synchronize_session=False
                )
            )

            db.session.flush()

            # -------------------------------------------------
            # Remove Issue header
            #
            # Bulk delete preserves the behavior of the
            # existing route implementation.
            # -------------------------------------------------

            deleted_issues = (
                ApplianceIssue.query
                .filter(
                    ApplianceIssue.id
                    == issue.id
                )
                .delete(
                    synchronize_session=False
                )
            )

            if deleted_issues != 1:
                raise ApplianceIssueError(
                    "Appliance Issue deletion failed."
                )

            db.session.commit()

            return {
                "deleted": True,
                "issue_id": issue_id_int,
                "issue_number": issue_number,
                "deleted_units": deleted_unit_count,
                "deleted_lines": int(
                    deleted_lines or 0
                ),
                "deleted_movements": int(
                    deleted_movements or 0
                ),
                "original_issue_movement_ids":
                    original_movement_ids,
            }

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

        _authorize_mutation("appliance.issue.return")

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

    # =========================================================
    # Remove one incorrectly-added appliance from Issue
    # =========================================================

    @staticmethod
    def remove_line_from_issue(
        *,
        actor: User,
        issue_line_id: int,
    ) -> dict:
        """
        Correction operation for SUPERADMIN.

        REMOVE FROM ISSUE means that this physical appliance
        should not have been included in this AIS at all.

        This is different from RETURN TO STOCK.

        Allowed only when no later EXTERNAL movement exists
        after this Issue's own movement chain.

        Same-Issue movements such as:

            ISSUE
            RETURN_TO_STOCK

        may be removed together because they belong to the
        correction chain being deleted.

        If this is the final line of the AIS, the empty
        ApplianceIssue header is also deleted.

        One transaction.
        """

        actor = (
            ApplianceIssueService
            ._require_user(actor)
        )

        role = (
            getattr(
                actor,
                "role",
                "",
            )
            or ""
        ).strip().lower()

        if role != "superadmin":
            raise ApplianceIssueAccessDenied(
                "Only SUPERADMIN can remove "
                "an appliance from an Issue."
            )

        try:
            line_id = int(
                issue_line_id
            )
        except (
            TypeError,
            ValueError,
        ):
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

        unit = db.session.get(
            ApplianceUnit,
            line.appliance_unit_id,
        )

        if unit is None:
            raise ApplianceIssueError(
                "Physical appliance not found."
            )

        line_status = (
            line.status
            or ""
        ).strip().lower()

        if line_status not in (
            "issued",
            "returned",
        ):
            raise ApplianceIssueError(
                f"{unit.inventory_number} has Issue status "
                f"{line_status.upper() or 'UNKNOWN'}. "
                "REMOVE FROM ISSUE is no longer allowed."
            )

        # -----------------------------------------------------
        # Movements belonging specifically to THIS Issue line.
        # -----------------------------------------------------

        same_line_movements = (
            ApplianceMovement.query
            .filter(
                ApplianceMovement.appliance_unit_id
                == unit.id,

                ApplianceMovement.issue_id
                == issue.id,

                ApplianceMovement.issue_line_id
                == line.id,
            )
            .order_by(
                ApplianceMovement.created_at.asc(),
                ApplianceMovement.id.asc(),
            )
            .all()
        )

        if not same_line_movements:
            raise ApplianceIssueError(
                f"Movement history is missing for "
                f"{unit.inventory_number}. "
                "REMOVE FROM ISSUE is blocked."
            )

        latest_own_movement = (
            same_line_movements[-1]
        )

        # -----------------------------------------------------
        # Only a later movement OUTSIDE this AIS blocks removal.
        #
        # Old movements before this AIS do NOT matter.
        # Movements belonging to this same AIS do NOT matter.
        # -----------------------------------------------------

        later_external_movement = (
            ApplianceMovement.query
            .filter(
                ApplianceMovement.appliance_unit_id
                == unit.id,

                db.or_(
                    ApplianceMovement.created_at
                    > latest_own_movement.created_at,

                    (
                        ApplianceMovement.created_at
                        == latest_own_movement.created_at
                    )
                    & (
                        ApplianceMovement.id
                        > latest_own_movement.id
                    ),
                ),

                db.or_(
                    ApplianceMovement.issue_id.is_(None),
                    ApplianceMovement.issue_id
                    != issue.id,
                ),
            )
            .order_by(
                ApplianceMovement.created_at.asc(),
                ApplianceMovement.id.asc(),
            )
            .first()
        )

        if later_external_movement is not None:
            raise ApplianceIssueError(
                f"{unit.inventory_number} already has later "
                f"movement "
                f"{later_external_movement.movement_type}. "
                "This Issue line can no longer be removed."
            )

        unit_status = (
            unit.status
            or ""
        ).strip().lower()

        # -----------------------------------------------------
        # Current physical state validation
        # -----------------------------------------------------

        if (
            line_status == "issued"
            and unit_status != "issued"
        ):
            raise ApplianceIssueError(
                f"{unit.inventory_number} is currently "
                f"{unit_status.upper() or 'UNKNOWN'}. "
                "REMOVE FROM ISSUE is blocked."
            )

        if (
            line_status == "returned"
            and unit_status != "available"
        ):
            raise ApplianceIssueError(
                f"{unit.inventory_number} was returned from "
                "this Issue but is no longer AVAILABLE. "
                "REMOVE FROM ISSUE is blocked."
            )

        issue_id = issue.id
        issue_number = issue.issue_number
        inventory_number = (
            unit.inventory_number
        )

        now = datetime.utcnow()

        try:

            # -------------------------------------------------
            # If line is still ISSUED, removing the erroneous
            # Issue must put physical appliance back AVAILABLE.
            #
            # If already RETURNED, it is already AVAILABLE.
            # -------------------------------------------------

            if line_status == "issued":

                unit.status = (
                    ApplianceIssueService
                    .STATUS_AVAILABLE
                )

                unit.current_work_order_id = None
                unit.current_work_order_number = None

                unit.updated_at = now
                unit.updated_by_id = actor.id

            # Returned line should already be available.
            # Clear current W/O defensively.
            else:

                unit.current_work_order_id = None
                unit.current_work_order_number = None

                unit.updated_at = now
                unit.updated_by_id = actor.id

            db.session.flush()

            # -------------------------------------------------
            # Remove ONLY this line's own movement chain.
            # -------------------------------------------------

            (
                ApplianceMovement.query
                .filter(
                    ApplianceMovement.issue_id
                    == issue.id,

                    ApplianceMovement.issue_line_id
                    == line.id,
                )
                .delete(
                    synchronize_session=False
                )
            )

            (
                ApplianceIssueLine.query
                .filter(
                    ApplianceIssueLine.id
                    == line.id
                )
                .delete(
                    synchronize_session=False
                )
            )

            db.session.flush()

            # -------------------------------------------------
            # Remaining lines?
            # -------------------------------------------------

            remaining_lines = (
                ApplianceIssueLine.query
                .filter(
                    ApplianceIssueLine.issue_id
                    == issue.id
                )
                .order_by(
                    ApplianceIssueLine.line_no.asc()
                )
                .all()
            )

            issue_deleted = False

            if not remaining_lines:

                (
                    ApplianceIssue.query
                    .filter(
                        ApplianceIssue.id
                        == issue.id
                    )
                    .delete(
                        synchronize_session=False
                    )
                )

                issue_deleted = True

            else:

                # ---------------------------------------------
                # Renumber safely: 1,2,3...
                #
                # Temporary negative numbers avoid violating
                # UNIQUE(issue_id, line_no).
                # ---------------------------------------------

                for index, remaining in enumerate(
                    remaining_lines,
                    start=1,
                ):
                    remaining.line_no = (
                        -100000 - index
                    )

                db.session.flush()

                for index, remaining in enumerate(
                    remaining_lines,
                    start=1,
                ):
                    remaining.line_no = index
                    remaining.updated_at = now

                issue.updated_at = now

            db.session.commit()

            return {
                "issue_id":
                    issue_id,

                "issue_number":
                    issue_number,

                "inventory_number":
                    inventory_number,

                "issue_deleted":
                    issue_deleted,
            }

        except Exception:
            db.session.rollback()
            raise

    # =========================================================
    # Replace one issued physical appliance
    # =========================================================

    @staticmethod
    def replace_appliance(
        *,
        actor: User,
        issue_line_id: int,
        new_appliance_unit_id: int,
        notes: str | None = None,
    ) -> dict:
        """
        Replace ONE currently issued physical appliance with
        another AVAILABLE physical appliance.

        The original Issue history is preserved.

        OLD:
            ApplianceUnit -> AVAILABLE
            IssueLine     -> REPLACED
            Movement      -> REPLACE_OUT

        NEW:
            ApplianceUnit -> ISSUED
            new IssueLine -> ISSUED
            Movement      -> REPLACE_IN

        Technician and W/O remain the same.

        Replacement is allowed only with the same appliance
        category/type.

        One atomic transaction.
        """

        actor = (
            ApplianceIssueService
            ._require_user(actor)
        )

        # -----------------------------------------------------
        # Normalize IDs
        # -----------------------------------------------------

        try:
            line_id = int(
                issue_line_id
            )

            new_unit_id = int(
                new_appliance_unit_id
            )

        except (
            TypeError,
            ValueError,
        ):
            raise ApplianceIssueError(
                "Invalid appliance replacement."
            )

        # -----------------------------------------------------
        # Load original Issue Line
        # -----------------------------------------------------

        old_line = db.session.get(
            ApplianceIssueLine,
            line_id,
        )

        if old_line is None:
            raise ApplianceIssueError(
                "Appliance Issue line not found."
            )

        issue = db.session.get(
            ApplianceIssue,
            old_line.issue_id,
        )

        if issue is None:
            raise ApplianceIssueError(
                "Appliance Issue not found."
            )

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
        # Old physical appliance
        # -----------------------------------------------------

        old_unit = db.session.get(
            ApplianceUnit,
            old_line.appliance_unit_id,
        )

        if old_unit is None:
            raise ApplianceIssueError(
                "Original physical appliance not found."
            )

        old_line_status = (
            old_line.status
            or ""
        ).strip().lower()

        if (
            old_line_status
            != ApplianceIssueService.LINE_STATUS_ISSUED
        ):
            raise ApplianceIssueError(
                f"{old_unit.inventory_number} cannot be "
                "replaced because this Issue line is "
                f"{old_line_status.upper() or 'UNKNOWN'}."
            )

        old_unit_status = (
            old_unit.status
            or ""
        ).strip().lower()

        if (
            old_unit_status
            != ApplianceIssueService.STATUS_ISSUED
        ):
            raise ApplianceIssueError(
                f"{old_unit.inventory_number} cannot be "
                "replaced because its inventory status is "
                f"{old_unit_status.upper() or 'UNKNOWN'}."
            )

        # -----------------------------------------------------
        # New physical appliance
        # -----------------------------------------------------

        new_unit = db.session.get(
            ApplianceUnit,
            new_unit_id,
        )

        if new_unit is None:
            raise ApplianceIssueError(
                "Replacement appliance not found."
            )

        if int(new_unit.id) == int(old_unit.id):
            raise ApplianceIssueError(
                "Select a different appliance."
            )

        if (
            int(new_unit.warehouse_id)
            != int(warehouse.id)
        ):
            raise ApplianceIssueError(
                f"{new_unit.inventory_number} belongs "
                "to another warehouse."
            )

        new_unit_status = (
            new_unit.status
            or ""
        ).strip().lower()

        if (
            new_unit_status
            != ApplianceIssueService.STATUS_AVAILABLE
        ):
            raise ApplianceIssueError(
                f"{new_unit.inventory_number} is not "
                f"AVAILABLE. Current status: "
                f"{new_unit_status.upper() or 'UNKNOWN'}."
            )

        # -----------------------------------------------------
        # Same appliance type/category only
        # -----------------------------------------------------

        if (
            int(new_unit.category_id)
            != int(old_unit.category_id)
        ):
            old_type = (
                old_line.appliance_type_snapshot
                or "UNKNOWN"
            ).upper()

            new_type = (
                new_unit.category.name
                if new_unit.category is not None
                else "UNKNOWN"
            ).upper()

            raise ApplianceIssueError(
                f"Replacement type must match. "
                f"Original: {old_type}. "
                f"Selected: {new_type}."
            )

        clean_notes = (
            ApplianceIssueService._clean_text(
                notes,
                max_length=2000,
            )
        )

        current_work_order_id = (
            old_line.current_work_order_id
        )

        current_work_order_number = (
            ApplianceIssueService._clean_text(
                old_line.current_work_order_number
                or issue.work_order_number,
                upper=True,
                max_length=120,
            )
        )

        now = datetime.utcnow()

        # -----------------------------------------------------
        # Next historical line number
        # -----------------------------------------------------

        max_line_no = (
            db.session.query(
                func.coalesce(
                    func.max(
                        ApplianceIssueLine.line_no
                    ),
                    0,
                )
            )
            .filter(
                ApplianceIssueLine.issue_id
                == issue.id
            )
            .scalar()
            or 0
        )

        new_line_no = int(
            max_line_no
        ) + 1

        try:

            # =================================================
            # 1. OLD APPLIANCE:
            #    ISSUED -> AVAILABLE
            # =================================================

            old_updated = (
                db.session.query(
                    ApplianceUnit
                )
                .filter(
                    ApplianceUnit.id
                    == old_unit.id,

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

            if old_updated != 1:
                raise ApplianceIssueError(
                    f"{old_unit.inventory_number} was changed "
                    "by another user. Nothing was replaced."
                )

            # =================================================
            # 2. NEW APPLIANCE:
            #    AVAILABLE -> ISSUED
            # =================================================

            new_updated = (
                db.session.query(
                    ApplianceUnit
                )
                .filter(
                    ApplianceUnit.id
                    == new_unit.id,

                    ApplianceUnit.warehouse_id
                    == warehouse.id,

                    ApplianceUnit.status
                    == ApplianceIssueService.STATUS_AVAILABLE,
                )
                .update(
                    {
                        ApplianceUnit.status:
                            ApplianceIssueService
                            .STATUS_ISSUED,

                        ApplianceUnit.current_work_order_id:
                            current_work_order_id,

                        ApplianceUnit.current_work_order_number:
                            current_work_order_number,

                        ApplianceUnit.updated_at:
                            now,

                        ApplianceUnit.updated_by_id:
                            actor.id,
                    },
                    synchronize_session=False,
                )
            )

            if new_updated != 1:
                raise ApplianceIssueError(
                    f"{new_unit.inventory_number} was changed "
                    "by another user. Nothing was replaced."
                )

            # =================================================
            # 3. Close old Issue Line
            # =================================================

            old_line.status = (
                ApplianceIssueService
                .LINE_STATUS_REPLACED
            )

            old_line.resolved_at = now
            old_line.updated_at = now

            if clean_notes:
                old_line.notes = clean_notes

            # =================================================
            # 4. Create new active Issue Line
            # =================================================

            new_category_name = (
                new_unit.category.name
                if new_unit.category is not None
                else None
            )

            new_line = ApplianceIssueLine(
                issue_id=issue.id,

                line_no=new_line_no,

                appliance_unit_id=new_unit.id,

                current_work_order_id=(
                    current_work_order_id
                ),

                current_work_order_number=(
                    current_work_order_number
                ),

                status=(
                    ApplianceIssueService
                    .LINE_STATUS_ISSUED
                ),

                installed_at=None,
                resolved_at=None,
                notes=None,

                inventory_number_snapshot=(
                    new_unit.inventory_number
                ),

                appliance_type_snapshot=(
                    new_category_name
                ),

                brand_snapshot=(
                    new_unit.brand
                ),

                model_number_snapshot=(
                    new_unit.model_number
                ),

                serial_number_snapshot=(
                    new_unit.serial_number
                ),

                size_value_snapshot=(
                    new_unit.size_value
                ),

                size_unit_snapshot=(
                    new_unit.size_unit
                ),

                condition_snapshot=(
                    new_unit.condition
                ),

                created_at=now,
                updated_at=now,
            )

            db.session.add(
                new_line
            )

            db.session.flush()

            # =================================================
            # 5. OLD -> warehouse movement
            # =================================================

            replace_out = ApplianceMovement(
                appliance_unit_id=old_unit.id,

                movement_type=(
                    ApplianceIssueService
                    .MOVEMENT_REPLACE_OUT
                ),

                issue_id=issue.id,
                issue_line_id=old_line.id,

                from_warehouse_id=None,
                to_warehouse_id=warehouse.id,

                from_work_order_id=(
                    current_work_order_id
                ),
                to_work_order_id=None,

                from_work_order_number=(
                    current_work_order_number
                ),
                to_work_order_number=None,

                technician_id=(
                    issue.technician_id
                ),

                related_appliance_unit_id=(
                    new_unit.id
                ),

                reason_code="APPLIANCE_REPLACEMENT",

                notes=clean_notes,

                meta_json=json.dumps(
                    {
                        "issue_number":
                            issue.issue_number,

                        "old_inventory_number":
                            old_unit.inventory_number,

                        "new_inventory_number":
                            new_unit.inventory_number,

                        "old_serial_number":
                            old_line.serial_number_snapshot,

                        "new_serial_number":
                            new_unit.serial_number,

                        "work_order_number":
                            current_work_order_number,

                        "warehouse_code":
                            warehouse.code,

                        "technician_username":
                            issue.technician_username,
                    },
                    ensure_ascii=False,
                ),

                actor_id=actor.id,
                created_at=now,
            )

            db.session.add(
                replace_out
            )

            # =================================================
            # 6. Warehouse -> technician/W/O movement
            # =================================================

            replace_in = ApplianceMovement(
                appliance_unit_id=new_unit.id,

                movement_type=(
                    ApplianceIssueService
                    .MOVEMENT_REPLACE_IN
                ),

                issue_id=issue.id,
                issue_line_id=new_line.id,

                from_warehouse_id=warehouse.id,
                to_warehouse_id=None,

                from_work_order_id=None,
                to_work_order_id=(
                    current_work_order_id
                ),

                from_work_order_number=None,
                to_work_order_number=(
                    current_work_order_number
                ),

                technician_id=(
                    issue.technician_id
                ),

                related_appliance_unit_id=(
                    old_unit.id
                ),

                reason_code="APPLIANCE_REPLACEMENT",

                notes=clean_notes,

                meta_json=json.dumps(
                    {
                        "issue_number":
                            issue.issue_number,

                        "old_inventory_number":
                            old_unit.inventory_number,

                        "new_inventory_number":
                            new_unit.inventory_number,

                        "old_serial_number":
                            old_line.serial_number_snapshot,

                        "new_serial_number":
                            new_unit.serial_number,

                        "work_order_number":
                            current_work_order_number,

                        "warehouse_code":
                            warehouse.code,

                        "technician_username":
                            issue.technician_username,
                    },
                    ensure_ascii=False,
                ),

                actor_id=actor.id,
                created_at=now,
            )

            db.session.add(
                replace_in
            )

            issue.updated_at = now

            # =================================================
            # ONE COMMIT
            # =================================================

            db.session.commit()

            db.session.refresh(
                new_line
            )

            return {
                "issue_id":
                    issue.id,

                "issue_number":
                    issue.issue_number,

                "old_line_id":
                    old_line.id,

                "new_line_id":
                    new_line.id,

                "old_inventory_number":
                    old_unit.inventory_number,

                "new_inventory_number":
                    new_unit.inventory_number,
            }

        except Exception:

            db.session.rollback()
            raise

    # =========================================================
    # Physical appliance disposition / status workflow
    # =========================================================

    @staticmethod
    def change_inventory_disposition(
        *,
        actor: User,
        appliance_unit_id: int,
        action: str,
        reason_code: str | None = None,
        notes: str | None = None,
        extra_movement_meta: dict | None = None,
    ) -> ApplianceUnit:
        """
        Change warehouse operational status for ONE physical
        appliance.

        This is NOT an Issue Slip operation.

        Supported transitions:

            AVAILABLE
                -> REPAIR
                -> VENDOR_RETURN_PENDING
                -> WRITTEN_OFF

            REPAIR
                -> AVAILABLE

            VENDOR_RETURN_PENDING
                -> VENDOR_RETURN
                -> AVAILABLE

        ISSUED appliances cannot use this workflow.
        They must first use RETURN / REPLACE / CHANGE W/O.

        Every transition creates immutable ApplianceMovement.
        """

        _authorize_mutation("appliance.inventory.disposition")

        actor = (
            ApplianceIssueService
            ._require_user(actor)
        )

        try:
            unit_id = int(
                appliance_unit_id
            )
        except (
            TypeError,
            ValueError,
        ):
            raise ApplianceIssueError(
                "Invalid appliance."
            )

        unit = db.session.get(
            ApplianceUnit,
            unit_id,
        )

        if unit is None:
            raise ApplianceIssueError(
                "Appliance not found."
            )

        warehouse = (
            ApplianceIssueService
            ._get_warehouse(
                unit.warehouse_id
            )
        )

        current_status = (
            unit.status
            or ""
        ).strip().lower()

        action_clean = (
            str(action or "")
            .strip()
            .upper()
        )

        reason_clean = (
            ApplianceIssueService._clean_text(
                reason_code,
                upper=True,
                max_length=80,
            )
        )

        notes_clean = (
            ApplianceIssueService._clean_text(
                notes,
                max_length=2000,
            )
        )

        transitions = {

            "SEND_TO_REPAIR": {
                "from":
                    ApplianceIssueService.STATUS_AVAILABLE,

                "to":
                    ApplianceIssueService.STATUS_REPAIR,

                "movement":
                    ApplianceIssueService.MOVEMENT_SEND_TO_REPAIR,

                "default_reason":
                    "NEEDS_REPAIR",
            },

                        "LOANER_RETURN_TO_STOCK_USED": {
                "from":
                    ApplianceIssueService.STATUS_AVAILABLE,

                "to":
                    ApplianceIssueService.STATUS_AVAILABLE,

                "movement":
                    ApplianceIssueService.MOVEMENT_VENDOR_RETURN_CANCEL,

                "default_reason":
                    "LOANER_RETURNED_TO_STOCK",

                "stock_class":
                    "new",

                "condition":
                    "used",
            },

            "LOANER_RETURN_TO_STOCK_REPAIRED": {
                "from":
                    ApplianceIssueService.STATUS_AVAILABLE,

                "to":
                    ApplianceIssueService.STATUS_AVAILABLE,

                "movement":
                    ApplianceIssueService.MOVEMENT_VENDOR_RETURN_CANCEL,

                "default_reason":
                    "LOANER_RETURNED_TO_STOCK",

                "stock_class":
                    "new",

                "condition":
                    "repaired",
            },

            "LOANER_RETURN_TO_STOCK_OPEN_BOX": {
                "from":
                    ApplianceIssueService.STATUS_AVAILABLE,

                "to":
                    ApplianceIssueService.STATUS_AVAILABLE,

                "movement":
                    ApplianceIssueService.MOVEMENT_VENDOR_RETURN_CANCEL,

                "default_reason":
                    "LOANER_RETURNED_TO_STOCK",

                "stock_class":
                    "new",

                "condition":
                    "open_box",
            },

            "LOANER_RETURN_TO_STOCK_NEW": {
                "from":
                    ApplianceIssueService.STATUS_AVAILABLE,

                "to":
                    ApplianceIssueService.STATUS_AVAILABLE,

                "movement":
                    ApplianceIssueService.MOVEMENT_VENDOR_RETURN_CANCEL,

                "default_reason":
                    "LOANER_RETURNED_TO_STOCK",

                "stock_class":
                    "new",

                "condition":
                    "new",
            },

            "LOANER_RETURN_TO_STOCK_DAMAGED": {
                "from":
                    ApplianceIssueService.STATUS_AVAILABLE,

                "to":
                    ApplianceIssueService.STATUS_AVAILABLE,

                "movement":
                    ApplianceIssueService.MOVEMENT_VENDOR_RETURN_CANCEL,

                "default_reason":
                    "LOANER_RETURNED_TO_STOCK",

                "stock_class":
                    "new",

                "condition":
                    "damaged",
            },



            "RETURN_FROM_REPAIR_TO_NEW": {
                "from":
                    ApplianceIssueService.STATUS_REPAIR,

                "to":
                    ApplianceIssueService.STATUS_AVAILABLE,

                "movement":
                    ApplianceIssueService.MOVEMENT_RETURN_FROM_REPAIR,

                "default_reason":
                    "NO_REPAIR_REQUIRED",

                "stock_class":
                    "new",

                "condition":
                    "new",
            },

            "RETURN_FROM_REPAIR_TO_LOANER": {
                "from":
                    ApplianceIssueService.STATUS_REPAIR,

                "to":
                    ApplianceIssueService.STATUS_AVAILABLE,

                "movement":
                    ApplianceIssueService.MOVEMENT_RETURN_FROM_REPAIR,

                "default_reason":
                    "REPAIR_COMPLETED",

                "stock_class":
                    "loaner",

                "condition":
                    "repaired",
            },

            "REPAIR_TO_VENDOR_RETURN": {
                "from":
                    ApplianceIssueService.STATUS_REPAIR,

                "to":
                    ApplianceIssueService.STATUS_VENDOR_RETURN_PENDING,

                "movement":
                    ApplianceIssueService.MOVEMENT_VENDOR_RETURN_PENDING,

                "default_reason":
                    "REPAIR_FAILED_VENDOR_RETURN",
            },

            "REPAIR_TO_SCRAP": {
                "from":
                    ApplianceIssueService.STATUS_REPAIR,

                "to":
                    ApplianceIssueService.STATUS_WRITTEN_OFF,

                "movement":
                    ApplianceIssueService.MOVEMENT_SCRAP,

                "default_reason":
                    "UNREPAIRABLE",
            },

            "MARK_VENDOR_RETURN": {
                "from":
                    ApplianceIssueService.STATUS_AVAILABLE,

                "to":
                    ApplianceIssueService.STATUS_VENDOR_RETURN_PENDING,

                "movement":
                    ApplianceIssueService.MOVEMENT_VENDOR_RETURN_PENDING,

                "default_reason":
                    "DEFECTIVE",
            },

            "CONFIRM_VENDOR_RETURN": {
                "from":
                    ApplianceIssueService.STATUS_VENDOR_RETURN_PENDING,

                "to":
                    ApplianceIssueService.STATUS_VENDOR_RETURN,

                "movement":
                    ApplianceIssueService.MOVEMENT_VENDOR_RETURN,

                "default_reason":
                    "RETURNED_TO_VENDOR",
            },

            "CANCEL_VENDOR_RETURN": {
                "from":
                    ApplianceIssueService.STATUS_VENDOR_RETURN_PENDING,

                "to":
                    ApplianceIssueService.STATUS_AVAILABLE,

                "movement":
                    ApplianceIssueService.MOVEMENT_VENDOR_RETURN_CANCEL,

                "default_reason":
                    "RETURN_CANCELLED",
            },

            "VENDOR_RETURN_TO_LOANER": {
                "from":
                    ApplianceIssueService.STATUS_VENDOR_RETURN_PENDING,

                "to":
                    ApplianceIssueService.STATUS_AVAILABLE,

                "movement":
                    ApplianceIssueService.MOVEMENT_VENDOR_RETURN_CANCEL,

                "default_reason":
                    "RETURNED_TO_STOCK_AS_LOANER",

                "stock_class":
                    "loaner",

                "condition":
                    "used",
            },

            "VENDOR_RETURN_TO_SCRAP": {
                "from":
                    ApplianceIssueService.STATUS_VENDOR_RETURN_PENDING,

                "to":
                    ApplianceIssueService.STATUS_WRITTEN_OFF,

                "movement":
                    ApplianceIssueService.MOVEMENT_SCRAP,

                "default_reason":
                    "VENDOR_RETURN_REJECTED_WRITE_OFF",
            },

            "SCRAP": {
                "from":
                    ApplianceIssueService.STATUS_AVAILABLE,

                "to":
                    ApplianceIssueService.STATUS_WRITTEN_OFF,

                "movement":
                    ApplianceIssueService.MOVEMENT_SCRAP,

                "default_reason":
                    "UNREPAIRABLE",
            },
        }

        rule = transitions.get(
            action_clean
        )

        if rule is None:
            raise ApplianceIssueError(
                "Unsupported appliance action."
            )

        # ----------------------------------------------------
        # Authorization belongs to the service.
        #
        # Routes and UI are not trusted to choose the correct
        # permission for a disposition action.
        #
        # Keep legacy appliance.receive behavior for existing
        # Repair / Loaner disposition workflows until those
        # workflows receive their own dedicated permission.
        # ----------------------------------------------------

        vendor_return_actions = {
            "MARK_VENDOR_RETURN",
            "REPAIR_TO_VENDOR_RETURN",
            "CONFIRM_VENDOR_RETURN",
            "CANCEL_VENDOR_RETURN",
            "VENDOR_RETURN_TO_LOANER",
            "VENDOR_RETURN_TO_SCRAP",
        }

        if action_clean in vendor_return_actions:
            required_permission = "appliance.vendor_return"
        elif action_clean == "SCRAP":
            required_permission = "appliance.write_off"
        else:
            required_permission = (
                ApplianceIssueService.ISSUE_PERMISSION
            )

        allowed = AccessControlService.can(
            actor,
            required_permission,
            warehouse_id=warehouse.id,
        )

        if not allowed:
            raise ApplianceIssueAccessDenied(
                "You do not have permission to perform "
                f"{action_clean.replace('_', ' ')} "
                f"for warehouse {warehouse.code}."
            )

        if current_status != rule["from"]:

            if current_status == "issued":

                raise ApplianceIssueError(
                    f"{unit.inventory_number} is currently ISSUED. "
                    "Use RETURN TO STOCK / REPLACE / CHANGE W/O first."
                )

            raise ApplianceIssueError(
                f"{unit.inventory_number} cannot perform "
                f"{action_clean.replace('_', ' ')} while "
                f"status is "
                f"{current_status.upper() or 'UNKNOWN'}."
            )

        new_status = rule["to"]

        reason_final = (
            reason_clean
            or rule["default_reason"]
        )

        # ----------------------------------------------------
        # Final immutable movement metadata.
        #
        # The service owns system/audit fields. Callers may
        # provide additional business metadata, but may not
        # overwrite service-owned keys.
        #
        # Build this BEFORE the inventory UPDATE so invalid
        # metadata fails without changing physical inventory.
        # ----------------------------------------------------

        movement_meta = {
            "inventory_number":
                unit.inventory_number,

            "action":
                action_clean,

            "from_status":
                current_status,

            "to_status":
                new_status,

            "old_stock_class":
                getattr(
                    unit,
                    "stock_class",
                    None,
                ),

            "new_stock_class":
                rule.get(
                    "stock_class"
                ),

            "old_condition":
                getattr(
                    unit,
                    "condition",
                    None,
                ),

            "new_condition":
                rule.get(
                    "condition"
                ),

            "warehouse_id":
                warehouse.id,

            "warehouse_code":
                warehouse.code,

            "reason_code":
                reason_final,
        }

        # ----------------------------------------------------
        # SEND_TO_REPAIR custody metadata.
        #
        # A physical appliance must not enter REPAIR without
        # recording who / what repair provider has custody.
        #
        # Validate here in the service, not only in the UI,
        # so future/direct callers cannot bypass the rule.
        # This block runs before the inventory UPDATE.
        # ----------------------------------------------------

        repair_number = None
        repair_type = None
        repair_vendor = None
        repair_technician_id = None
        repair_technician_username = None
        repair_reference = None

        if action_clean == "SEND_TO_REPAIR":

            if extra_movement_meta is None:
                extra_movement_meta = {}

            if not isinstance(
                extra_movement_meta,
                dict,
            ):
                raise ApplianceIssueError(
                    "Invalid movement metadata."
                )

            repair_type = str(
                extra_movement_meta.get(
                    "repair_type"
                )
                or ""
            ).strip().lower()

            repair_vendor = str(
                extra_movement_meta.get(
                    "repair_vendor"
                )
                or ""
            ).strip()

            repair_technician_raw = str(
                extra_movement_meta.get(
                    "repair_technician_id"
                )
                or ""
            ).strip()

            repair_reference = str(
                extra_movement_meta.get(
                    "repair_reference"
                )
                or ""
            ).strip()

            if repair_type not in (
                "vendor",
                "internal",
            ):
                raise ApplianceIssueError(
                    "Repair Performed By must be "
                    "Vendor or Internal Technician."
                )

            if len(repair_vendor) > 160:
                raise ApplianceIssueError(
                    "Repair Vendor is too long "
                    "(maximum 160 characters)."
                )

            if len(repair_reference) > 120:
                raise ApplianceIssueError(
                    "Repair Ref / Ticket is too long "
                    "(maximum 120 characters)."
                )

            if repair_type == "vendor":

                if not repair_vendor:
                    raise ApplianceIssueError(
                        "Repair Vendor is required."
                    )

                if repair_technician_raw:
                    raise ApplianceIssueError(
                        "Vendor repair cannot also have "
                        "an Internal Technician."
                    )

                repair_technician_id = None
                repair_technician_username = None

            else:

                if repair_vendor:
                    raise ApplianceIssueError(
                        "Internal repair cannot also have "
                        "a Repair Vendor."
                    )

                if not repair_technician_raw:
                    raise ApplianceIssueError(
                        "Internal Technician is required."
                    )

                try:
                    repair_technician_id = int(
                        repair_technician_raw
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    raise ApplianceIssueError(
                        "Invalid Internal Technician."
                    )

                repair_technician = db.session.get(
                    User,
                    repair_technician_id,
                )

                if repair_technician is None:
                    raise ApplianceIssueError(
                        "Internal Technician was not found."
                    )

                technician_role = (
                    getattr(
                        repair_technician,
                        "role",
                        "",
                    )
                    or ""
                ).strip().lower()

                if technician_role != "technician":
                    raise ApplianceIssueError(
                        "Selected user is not a technician."
                    )

                repair_technician_username = (
                    getattr(
                        repair_technician,
                        "username",
                        "",
                    )
                    or ""
                ).strip()

                if not repair_technician_username:
                    raise ApplianceIssueError(
                        "Internal Technician has no username."
                    )

                repair_vendor = None

            # Never mutate caller-owned metadata.
            extra_movement_meta = dict(
                extra_movement_meta
            )

            # Remove raw UI values first, then write only
            # normalized immutable repair metadata.
            extra_movement_meta.pop(
                "repair_vendor",
                None,
            )

            extra_movement_meta.pop(
                "repair_technician_id",
                None,
            )

            extra_movement_meta["repair_type"] = (
                repair_type
            )

            extra_movement_meta["repair_vendor"] = (
                repair_vendor
            )

            extra_movement_meta[
                "repair_technician_id"
            ] = repair_technician_id

            extra_movement_meta[
                "repair_technician_username"
            ] = repair_technician_username

            extra_movement_meta[
                "repair_reference"
            ] = repair_reference

            # Internal WCCR Repair Order number.
            repair_number = (
                ApplianceIssueService
                ._next_repair_number()
            )

            extra_movement_meta[
                "repair_number"
            ] = repair_number

        if extra_movement_meta is not None:

            if not isinstance(
                extra_movement_meta,
                dict,
            ):
                raise ApplianceIssueError(
                    "Invalid movement metadata."
                )

            protected_keys = set(
                movement_meta.keys()
            )

            conflicting_keys = sorted(
                protected_keys.intersection(
                    extra_movement_meta.keys()
                )
            )

            if conflicting_keys:
                raise ApplianceIssueError(
                    "Movement metadata cannot overwrite "
                    "system audit fields: "
                    + ", ".join(conflicting_keys)
                )

            movement_meta.update(
                extra_movement_meta
            )

        # ----------------------------------------------------
        # Repair Order lifecycle closure.
        #
        # Any action that moves an appliance OUT of REPAIR
        # must close exactly one existing OPEN Repair Order.
        #
        # Resolve and validate this BEFORE the physical
        # ApplianceUnit UPDATE so broken repair history can
        # never leave inventory partially changed.
        # ----------------------------------------------------

        repair_order_to_close = None
        repair_close_status = None
        repair_close_outcome = None

        repair_close_rules = {
            "RETURN_FROM_REPAIR_TO_NEW": {
                "status": "completed",
                "outcome": "returned_to_new",
            },

            "RETURN_FROM_REPAIR_TO_LOANER": {
                "status": "completed",
                "outcome": "returned_to_loaner",
            },

            "REPAIR_TO_VENDOR_RETURN": {
                "status": "vendor_return",
                "outcome": "vendor_return",
            },

            "REPAIR_TO_SCRAP": {
                "status": "scrapped",
                "outcome": "scrapped",
            },
        }

        repair_close_rule = repair_close_rules.get(
            action_clean
        )

        if repair_close_rule is not None:

            open_repair_orders = (
                ApplianceRepairOrder.query
                .filter(
                    ApplianceRepairOrder.appliance_unit_id
                    == unit.id,

                    ApplianceRepairOrder.status
                    == "open",
                )
                .order_by(
                    ApplianceRepairOrder.id.asc()
                )
                .all()
            )

            if len(open_repair_orders) == 0:
                raise ApplianceIssueError(
                    f"{unit.inventory_number} is in REPAIR "
                    "but has no OPEN Repair Order."
                )

            if len(open_repair_orders) > 1:
                repair_numbers = ", ".join(
                    str(order.repair_number)
                    for order in open_repair_orders
                )

                raise ApplianceIssueError(
                    f"{unit.inventory_number} has multiple "
                    f"OPEN Repair Orders: {repair_numbers}. "
                    "Repair history must be corrected first."
                )

            repair_order_to_close = (
                open_repair_orders[0]
            )

            if (
                repair_order_to_close.warehouse_id
                != warehouse.id
            ):
                raise ApplianceIssueError(
                    f"{unit.inventory_number} Repair Order "
                    "warehouse does not match the appliance "
                    "warehouse."
                )

            repair_close_status = (
                repair_close_rule["status"]
            )

            repair_close_outcome = (
                repair_close_rule["outcome"]
            )

            # These are service-owned audit fields for repair
            # outcome movements. A caller may not supply them.
            repair_audit_conflicts = [
                key
                for key in (
                    "repair_number",
                    "repair_order_id",
                )
                if key in movement_meta
            ]

            if repair_audit_conflicts:
                raise ApplianceIssueError(
                    "Movement metadata cannot overwrite "
                    "repair audit fields: "
                    + ", ".join(
                        repair_audit_conflicts
                    )
                )

            movement_meta["repair_number"] = (
                repair_order_to_close.repair_number
            )

            movement_meta["repair_order_id"] = (
                repair_order_to_close.id
            )

            movement_meta["repair_outcome"] = (
                repair_close_outcome
            )

        try:
            movement_meta_json = json.dumps(
                movement_meta,
                ensure_ascii=False,
            )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ApplianceIssueError(
                "Movement metadata is not JSON serializable."
            ) from exc

        now = datetime.utcnow()

        # ----------------------------------------------------
        # Atomic physical inventory update.
        #
        # Most transitions change STATUS only.
        # Repair outcomes may also intentionally change:
        #
        #   stock_class
        #   condition
        # ----------------------------------------------------

        update_values = {
            ApplianceUnit.status:
                new_status,

            ApplianceUnit.current_work_order_id:
                None,

            ApplianceUnit.current_work_order_number:
                None,

            ApplianceUnit.updated_at:
                now,

            ApplianceUnit.updated_by_id:
                actor.id,
        }

        if rule.get(
            "stock_class"
        ) is not None:

            update_values[
                ApplianceUnit.stock_class
            ] = rule[
                "stock_class"
            ]

        if rule.get(
            "condition"
        ) is not None:

            update_values[
                ApplianceUnit.condition
            ] = rule[
                "condition"
            ]

        try:

            updated = (
                db.session.query(
                    ApplianceUnit
                )
                .filter(
                    ApplianceUnit.id
                    == unit.id,

                    ApplianceUnit.status
                    == current_status,
                )
                .update(
                    update_values,
                    synchronize_session=False,
                )
            )

            if updated != 1:
                raise ApplianceIssueError(
                    f"{unit.inventory_number} was changed "
                    "by another user. Refresh and try again."
                )

            # ------------------------------------------------
            # First-class Repair Order.
            #
            # Transitional compatibility:
            # the current UI still supplies one text field
            # named repair_provider. Until the UI selector is
            # upgraded, that legacy field represents an
            # external vendor.
            #
            # The Repair Order, ApplianceUnit update and
            # ApplianceMovement all remain in ONE transaction.
            # ------------------------------------------------

            if action_clean == "SEND_TO_REPAIR":

                if not repair_number:
                    raise ApplianceIssueError(
                        "Internal Repair Order number "
                        "was not generated."
                    )

                repair_order = ApplianceRepairOrder(
                    repair_number=repair_number,

                    appliance_unit_id=unit.id,
                    warehouse_id=warehouse.id,

                    repair_type=repair_type,
                    repair_vendor=repair_vendor,
                    repair_technician_id=(
                        repair_technician_id
                    ),

                    provider_reference=(
                        repair_reference
                        or None
                    ),

                    status="open",

                    reason_code=reason_final,
                    notes=notes_clean,

                    sent_at=now,
                    sent_by_id=actor.id,

                    completed_at=None,
                    completed_by_id=None,
                    outcome=None,

                    created_at=now,
                    updated_at=now,
                )

                db.session.add(
                    repair_order
                )

            # ------------------------------------------------
            # Close the SAME Repair Order that was validated
            # before the ApplianceUnit UPDATE.
            #
            # This remains inside the same database
            # transaction as ApplianceUnit + Movement.
            # ------------------------------------------------

            if repair_order_to_close is not None:

                repair_order_to_close.status = (
                    repair_close_status
                )

                repair_order_to_close.outcome = (
                    repair_close_outcome
                )

                repair_order_to_close.completed_at = now
                repair_order_to_close.completed_by_id = (
                    actor.id
                )

                repair_order_to_close.updated_at = now

            movement = ApplianceMovement(
                appliance_unit_id=unit.id,

                movement_type=rule[
                    "movement"
                ],

                issue_id=None,
                issue_line_id=None,

                from_warehouse_id=(
                    warehouse.id
                ),

                to_warehouse_id=(
                    warehouse.id
                    if new_status
                    in (
                        ApplianceIssueService.STATUS_AVAILABLE,
                        ApplianceIssueService.STATUS_REPAIR,
                        ApplianceIssueService.STATUS_VENDOR_RETURN_PENDING,
                    )
                    else None
                ),

                from_work_order_id=None,
                to_work_order_id=None,

                from_work_order_number=None,
                to_work_order_number=None,

                technician_id=None,

                related_appliance_unit_id=None,

                reason_code=reason_final,

                notes=notes_clean,

                meta_json=movement_meta_json,

                actor_id=actor.id,
                created_at=now,
            )

            db.session.add(
                movement
            )

            db.session.commit()

            refreshed = db.session.get(
                ApplianceUnit,
                unit.id,
            )

            return refreshed

        except Exception:
            db.session.rollback()
            raise

