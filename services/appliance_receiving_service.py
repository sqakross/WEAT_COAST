from __future__ import annotations
from license_client.operation_gate import authorize_mutation as _authorize_mutation

from datetime import date, datetime
from decimal import Decimal, InvalidOperation

from sqlalchemy import func

from extensions import db
from models import (
    ApplianceCategory,
    ApplianceIssueLine,
    ApplianceMovement,
    ApplianceReceiving,
    ApplianceReceivingLine,
    ApplianceUnit,
    User,
    Warehouse,
)
from services.access_control_service import AccessControlService


class ApplianceReceivingError(Exception):
    """Expected appliance receiving business error."""


class ApplianceAccessDenied(ApplianceReceivingError):
    pass


class ApplianceReceivingNotFound(ApplianceReceivingError):
    pass


class ApplianceReceivingService:
    """
    Business service for Appliance Inventory Receiving.

    Transaction policy:
        Public write operations commit once on success and rollback
        completely on failure.

    Workflow:
        create draft
            -> add/edit/delete physical appliance rows
            -> post receiving
            -> create one ApplianceUnit per receiving line

    Pricing is intentionally independent from posting:
        unit_cost=None means manager still needs to enter the cost.

    Existing Parts / GoodsReceipt inventory is not touched.
    """

    STATUS_DRAFT = "draft"
    STATUS_POSTED = "posted"
    STATUS_VOIDED = "voided"

    UNIT_STATUS_AVAILABLE = "available"
    UNIT_STATUS_VOIDED = "voided"

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

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

        if max_length is not None and len(value) > max_length:
            raise ApplianceReceivingError(
                f"Value exceeds maximum length of {max_length}."
            )

        return value

    @staticmethod
    def _money_or_none(value) -> float | None:
        """
        None / blank -> None.

        Zero is a valid explicit price and remains 0.0.
        Negative prices are rejected.
        """
        if value is None:
            return None

        if isinstance(value, str):
            value = value.strip()

            if not value:
                return None

            value = value.replace(",", "")

        try:
            amount = Decimal(str(value))
        except (InvalidOperation, ValueError, TypeError):
            raise ApplianceReceivingError(
                f"Invalid monetary value: {value!r}"
            )

        if amount < 0:
            raise ApplianceReceivingError(
                "Price cannot be negative."
            )

        return float(
            amount.quantize(Decimal("0.01"))
        )

    @staticmethod
    def _float_or_none(value) -> float | None:
        if value is None:
            return None

        if isinstance(value, str):
            value = value.strip()

            if not value:
                return None

        try:
            result = float(value)
        except (TypeError, ValueError):
            raise ApplianceReceivingError(
                f"Invalid numeric value: {value!r}"
            )

        if result < 0:
            raise ApplianceReceivingError(
                "Size cannot be negative."
            )

        return result

    @staticmethod
    def _require_user(actor: User | None) -> User:
        if actor is None or getattr(actor, "id", None) is None:
            raise ApplianceAccessDenied(
                "Authenticated user is required."
            )

        return actor

    @staticmethod
    def _get_active_warehouse(
        warehouse_id: int,
    ) -> Warehouse:
        warehouse = Warehouse.query.get(warehouse_id)

        if warehouse is None or not warehouse.is_active:
            raise ApplianceReceivingError(
                "Warehouse not found or inactive."
            )

        return warehouse

    @staticmethod
    def _require_permission(
        *,
        actor: User,
        permission_code: str,
        warehouse_id: int,
    ) -> None:
        if not AccessControlService.can(
            actor,
            permission_code,
            warehouse_id=warehouse_id,
        ):
            raise ApplianceAccessDenied(
                f"Access denied: {permission_code}"
            )

    @staticmethod
    def _get_receiving(
        receiving_id: int,
    ) -> ApplianceReceiving:
        receiving = db.session.get(
            ApplianceReceiving,
            receiving_id,
        )

        if receiving is None:
            raise ApplianceReceivingNotFound(
                f"Receiving {receiving_id} not found."
            )

        return receiving

    @staticmethod
    def _require_draft(
        receiving: ApplianceReceiving,
    ) -> None:
        if receiving.status != ApplianceReceivingService.STATUS_DRAFT:
            raise ApplianceReceivingError(
                "Only Draft Receiving can be modified."
            )

    @staticmethod
    def _get_active_category(
        category_id: int,
    ) -> ApplianceCategory:
        category = db.session.get(
            ApplianceCategory,
            category_id,
        )

        if category is None or not category.is_active:
            raise ApplianceReceivingError(
                "Appliance category not found or inactive."
            )

        return category

    @staticmethod
    def _next_receiving_number() -> str:
        """
        Human-readable receiving number.

        The unique constraint remains the final duplicate protection.

        For the current single SQLite writer this is sufficient.
        """
        max_id = (
            db.session.query(
                func.coalesce(
                    func.max(ApplianceReceiving.id),
                    0,
                )
            )
            .scalar()
            or 0
        )

        return f"ARCV-{int(max_id) + 1:06d}"

    @staticmethod
    def _next_inventory_number() -> str:
        max_id = (
            db.session.query(
                func.coalesce(
                    func.max(ApplianceUnit.id),
                    0,
                )
            )
            .scalar()
            or 0
        )

        return f"AP-{int(max_id) + 1:07d}"

    @staticmethod
    def _next_line_number(
        receiving: ApplianceReceiving,
    ) -> int:
        max_line = (
            db.session.query(
                func.coalesce(
                    func.max(ApplianceReceivingLine.line_no),
                    0,
                )
            )
            .filter(
                ApplianceReceivingLine.receiving_id
                == receiving.id
            )
            .scalar()
            or 0
        )

        return int(max_line) + 1

    @staticmethod
    def _check_serial_duplicate(
        *,
        serial_number: str | None,
        exclude_line_id: int | None = None,
    ) -> None:
        """
        Serial is optional because warehouse staff may receive a unit
        whose serial cannot yet be read.

        If supplied, however, it must not already exist in another
        active receiving row or ApplianceUnit.
        """
        if not serial_number:
            return

        serial_key = serial_number.strip().upper()

        unit_exists = (
            ApplianceUnit.query
            .filter(
                func.upper(ApplianceUnit.serial_number)
                == serial_key
            )
            .first()
        )

        if unit_exists is not None:
            raise ApplianceReceivingError(
                f"Serial number already exists in stock: "
                f"{serial_number}"
            )

        line_query = (
            ApplianceReceivingLine.query
            .filter(
                func.upper(
                    ApplianceReceivingLine.serial_number
                )
                == serial_key
            )
        )

        if exclude_line_id is not None:
            line_query = line_query.filter(
                ApplianceReceivingLine.id
                != exclude_line_id
            )

        line_exists = line_query.first()

        if line_exists is not None:
            raise ApplianceReceivingError(
                f"Serial number already exists in Receiving: "
                f"{serial_number}"
            )

    # --------------------------------------------------------
    # Read operations
    # --------------------------------------------------------

    @staticmethod
    def get_receiving(
        *,
        receiving_id: int,
        actor: User,
        permission_code: str = "appliance.view",
    ) -> ApplianceReceiving:
        actor = ApplianceReceivingService._require_user(actor)

        receiving = ApplianceReceivingService._get_receiving(
            receiving_id
        )

        ApplianceReceivingService._require_permission(
            actor=actor,
            permission_code=permission_code,
            warehouse_id=receiving.warehouse_id,
        )

        return receiving

    @staticmethod
    def list_missing_price_receivings(
        *,
        actor: User,
        warehouse_id: int | None = None,
    ) -> list[ApplianceReceiving]:
        """
        Manager queue.

        Returns POSTED receivings containing at least one line where
        unit_cost IS NULL.

        This is the queue that will drive the Missing Prices alert/UI.
        """
        actor = ApplianceReceivingService._require_user(actor)

        query = (
            ApplianceReceiving.query
            .join(
                ApplianceReceivingLine,
                ApplianceReceivingLine.receiving_id
                == ApplianceReceiving.id,
            )
            .filter(
                ApplianceReceiving.status
                == ApplianceReceivingService.STATUS_POSTED,
                ApplianceReceivingLine.unit_cost.is_(None),
            )
            .distinct()
            .order_by(
                ApplianceReceiving.received_at.asc(),
                ApplianceReceiving.id.asc(),
            )
        )

        if warehouse_id is not None:
            ApplianceReceivingService._get_active_warehouse(
                warehouse_id
            )

            ApplianceReceivingService._require_permission(
                actor=actor,
                permission_code="appliance.pricing",
                warehouse_id=warehouse_id,
            )

            query = query.filter(
                ApplianceReceiving.warehouse_id
                == warehouse_id
            )

            return query.all()

        allowed_ids = [
            warehouse.id
            for warehouse in AccessControlService.accessible_warehouses(
                actor
            )
            if AccessControlService.can(
                actor,
                "appliance.pricing",
                warehouse_id=warehouse.id,
            )
        ]

        if not allowed_ids:
            return []

        return (
            query
            .filter(
                ApplianceReceiving.warehouse_id.in_(
                    allowed_ids
                )
            )
            .all()
        )

    # --------------------------------------------------------
    # Draft Receiving
    # --------------------------------------------------------

    @staticmethod
    def create_draft(
        *,
        actor: User,
        warehouse_id: int,
        supplier_name: str | None = None,
        invoice_number: str | None = None,
        invoice_date: date | None = None,
        received_at: datetime | None = None,
        notes: str | None = None,
        attachment_path: str | None = None,
    ) -> ApplianceReceiving:
        actor = ApplianceReceivingService._require_user(actor)

        warehouse = (
            ApplianceReceivingService._get_active_warehouse(
                warehouse_id
            )
        )

        ApplianceReceivingService._require_permission(
            actor=actor,
            permission_code="appliance.receive",
            warehouse_id=warehouse.id,
        )

        receiving = ApplianceReceiving(
            receiving_number=(
                ApplianceReceivingService
                ._next_receiving_number()
            ),
            warehouse_id=warehouse.id,
            supplier_name=(
                ApplianceReceivingService._clean_text(
                    supplier_name,
                    max_length=200,
                )
            ),
            invoice_number=(
                ApplianceReceivingService._clean_text(
                    invoice_number,
                    upper=True,
                    max_length=80,
                )
            ),
            invoice_date=invoice_date,
            received_at=received_at or datetime.utcnow(),
            status=ApplianceReceivingService.STATUS_DRAFT,
            notes=(
                ApplianceReceivingService._clean_text(
                    notes
                )
            ),
            attachment_path=(
                ApplianceReceivingService._clean_text(
                    attachment_path,
                    max_length=512,
                )
            ),
            created_by_id=actor.id,
            updated_by_id=actor.id,
        )

        try:
            db.session.add(receiving)
            db.session.commit()
            db.session.refresh(receiving)
            return receiving

        except Exception:
            db.session.rollback()
            raise

    @staticmethod
    def update_draft_header(
        *,
        receiving_id: int,
        actor: User,
        supplier_name: str | None = None,
        invoice_number: str | None = None,
        invoice_date: date | None = None,
        received_at: datetime | None = None,
        notes: str | None = None,
        attachment_path: str | None = None,
    ) -> ApplianceReceiving:
        actor = ApplianceReceivingService._require_user(actor)

        receiving = (
            ApplianceReceivingService._get_receiving(
                receiving_id
            )
        )

        ApplianceReceivingService._require_draft(receiving)

        ApplianceReceivingService._require_permission(
            actor=actor,
            permission_code="appliance.receive",
            warehouse_id=receiving.warehouse_id,
        )

        receiving.supplier_name = (
            ApplianceReceivingService._clean_text(
                supplier_name,
                max_length=200,
            )
        )

        receiving.invoice_number = (
            ApplianceReceivingService._clean_text(
                invoice_number,
                upper=True,
                max_length=80,
            )
        )

        receiving.invoice_date = invoice_date

        if received_at is not None:
            receiving.received_at = received_at

        receiving.notes = (
            ApplianceReceivingService._clean_text(notes)
        )

        receiving.attachment_path = (
            ApplianceReceivingService._clean_text(
                attachment_path,
                max_length=512,
            )
        )

        receiving.updated_by_id = actor.id
        receiving.updated_at = datetime.utcnow()

        try:
            db.session.commit()
            db.session.refresh(receiving)
            return receiving

        except Exception:
            db.session.rollback()
            raise

    @staticmethod
    def add_line(
        *,
        receiving_id: int,
        actor: User,
        category_id: int,
        brand: str | None = None,
        model_number: str | None = None,
        serial_number: str | None = None,
        description: str | None = None,
        size_value=None,
        size_unit: str | None = None,
        color: str | None = None,
        condition: str = "new",
        unit_cost=None,
        selling_price=None,
        notes: str | None = None,
    ) -> ApplianceReceivingLine:
        actor = ApplianceReceivingService._require_user(actor)

        receiving = (
            ApplianceReceivingService._get_receiving(
                receiving_id
            )
        )

        ApplianceReceivingService._require_draft(receiving)

        ApplianceReceivingService._require_permission(
            actor=actor,
            permission_code="appliance.receive",
            warehouse_id=receiving.warehouse_id,
        )

        ApplianceReceivingService._get_active_category(
            category_id
        )

        serial_clean = (
            ApplianceReceivingService._clean_text(
                serial_number,
                upper=True,
                max_length=160,
            )
        )

        ApplianceReceivingService._check_serial_duplicate(
            serial_number=serial_clean
        )

        cost = ApplianceReceivingService._money_or_none(
            unit_cost
        )

        price = ApplianceReceivingService._money_or_none(
            selling_price
        )

        # Entering financial values requires pricing permission.
        if cost is not None or price is not None:
            ApplianceReceivingService._require_permission(
                actor=actor,
                permission_code="appliance.pricing",
                warehouse_id=receiving.warehouse_id,
            )

        condition_clean = (
            ApplianceReceivingService._clean_text(
                condition,
                max_length=40,
            )
            or "new"
        ).lower()

        line = ApplianceReceivingLine(
            receiving_id=receiving.id,
            line_no=(
                ApplianceReceivingService._next_line_number(
                    receiving
                )
            ),
            category_id=category_id,
            brand=(
                ApplianceReceivingService._clean_text(
                    brand,
                    upper=True,
                    max_length=100,
                )
            ),
            model_number=(
                ApplianceReceivingService._clean_text(
                    model_number,
                    upper=True,
                    max_length=120,
                )
            ),
            serial_number=serial_clean,
            description=(
                ApplianceReceivingService._clean_text(
                    description,
                    max_length=500,
                )
            ),
            size_value=(
                ApplianceReceivingService._float_or_none(
                    size_value
                )
            ),
            size_unit=(
                ApplianceReceivingService._clean_text(
                    size_unit,
                    max_length=20,
                )
            ),
            color=(
                ApplianceReceivingService._clean_text(
                    color,
                    upper=True,
                    max_length=80,
                )
            ),
            condition=condition_clean,
            unit_cost=cost,
            selling_price=price,
            notes=(
                ApplianceReceivingService._clean_text(notes)
            ),
            created_by_id=actor.id,
            updated_by_id=actor.id,
        )

        receiving.updated_by_id = actor.id
        receiving.updated_at = datetime.utcnow()

        try:
            db.session.add(line)
            db.session.commit()
            db.session.refresh(line)
            return line

        except Exception:
            db.session.rollback()
            raise

    @staticmethod
    def add_lines_bulk(
        *,
        receiving_id: int,
        actor: User,
        rows: list[dict],
    ) -> list[ApplianceReceivingLine]:
        """
        Add multiple physical appliances to a Draft Receiving.

        Atomic behavior:
            validate entire payload
                -> insert all rows
                -> one commit

        If any row fails validation, nothing is saved.
        """
        actor = ApplianceReceivingService._require_user(actor)

        receiving = (
            ApplianceReceivingService._get_receiving(
                receiving_id
            )
        )

        ApplianceReceivingService._require_draft(receiving)

        ApplianceReceivingService._require_permission(
            actor=actor,
            permission_code="appliance.receive",
            warehouse_id=receiving.warehouse_id,
        )

        if not isinstance(rows, list):
            raise ApplianceReceivingError(
                "Rows must be a list."
            )

        if not rows:
            raise ApplianceReceivingError(
                "No appliance rows were provided."
            )

        if len(rows) > 250:
            raise ApplianceReceivingError(
                "Maximum 250 appliance rows per batch."
            )

        # ----------------------------------------------------
        # Validate category IDs in one query
        # ----------------------------------------------------

        requested_category_ids = set()

        for index, raw in enumerate(rows, 1):
            if not isinstance(raw, dict):
                raise ApplianceReceivingError(
                    f"Row {index}: invalid row format."
                )

            try:
                category_id = int(
                    raw.get("category_id") or 0
                )
            except (TypeError, ValueError):
                category_id = 0

            if not category_id:
                raise ApplianceReceivingError(
                    f"Row {index}: appliance category "
                    f"is required."
                )

            requested_category_ids.add(category_id)

        valid_categories = {
            int(row.id): row
            for row in (
                ApplianceCategory.query
                .filter(
                    ApplianceCategory.id.in_(
                        requested_category_ids
                    ),
                    ApplianceCategory.is_active.is_(True),
                )
                .all()
            )
        }

        missing_category_ids = (
            requested_category_ids
            - set(valid_categories)
        )

        if missing_category_ids:
            raise ApplianceReceivingError(
                "One or more appliance categories "
                "are invalid or inactive."
            )

        # ----------------------------------------------------
        # Normalize all rows before touching DB
        # ----------------------------------------------------

        normalized_rows = []
        payload_serials = set()

        financial_values_present = False

        for index, raw in enumerate(rows, 1):

            category_id = int(
                raw.get("category_id") or 0
            )

            serial_number = (
                ApplianceReceivingService._clean_text(
                    raw.get("serial_number"),
                    upper=True,
                    max_length=160,
                )
            )

            if serial_number:
                serial_key = serial_number.upper()

                if serial_key in payload_serials:
                    raise ApplianceReceivingError(
                        f"Row {index}: duplicate serial "
                        f"in this batch: {serial_number}"
                    )

                payload_serials.add(serial_key)

            unit_cost = (
                ApplianceReceivingService._money_or_none(
                    raw.get("unit_cost")
                )
            )

            selling_price = (
                ApplianceReceivingService._money_or_none(
                    raw.get("selling_price")
                )
            )

            if (
                unit_cost is not None
                or selling_price is not None
            ):
                financial_values_present = True

            condition = (
                ApplianceReceivingService._clean_text(
                    raw.get("condition"),
                    max_length=40,
                )
                or "new"
            ).lower()

            normalized_rows.append(
                {
                    "category_id": category_id,
                    "brand": (
                        ApplianceReceivingService._clean_text(
                            raw.get("brand"),
                            upper=True,
                            max_length=100,
                        )
                    ),
                    "model_number": (
                        ApplianceReceivingService._clean_text(
                            raw.get("model_number"),
                            upper=True,
                            max_length=120,
                        )
                    ),
                    "serial_number": serial_number,
                    "description": (
                        ApplianceReceivingService._clean_text(
                            raw.get("description"),
                            max_length=500,
                        )
                    ),
                    "size_value": (
                        ApplianceReceivingService._float_or_none(
                            raw.get("size_value")
                        )
                    ),
                    "size_unit": (
                        ApplianceReceivingService._clean_text(
                            raw.get("size_unit"),
                            max_length=20,
                        )
                    ),

                    "color": (
                        ApplianceReceivingService._clean_text(
                            raw.get("color"),
                            upper=True,
                            max_length=80,
                        )
                    ),
                    "condition": condition,
                    "unit_cost": unit_cost,
                    "selling_price": selling_price,
                    "notes": (
                        ApplianceReceivingService._clean_text(
                            raw.get("notes")
                        )
                    ),
                }
            )

        # ----------------------------------------------------
        # Financial access
        # ----------------------------------------------------

        if financial_values_present:
            ApplianceReceivingService._require_permission(
                actor=actor,
                permission_code="appliance.pricing",
                warehouse_id=receiving.warehouse_id,
            )

        # ----------------------------------------------------
        # Validate serials against existing Receiving rows
        # and current Appliance stock in two SQL queries.
        # ----------------------------------------------------

        if payload_serials:

            existing_line_serials = {
                str(serial).strip().upper()
                for (serial,) in (
                    db.session.query(
                        ApplianceReceivingLine.serial_number
                    )
                    .filter(
                        ApplianceReceivingLine.serial_number
                        .isnot(None),
                        func.upper(
                            ApplianceReceivingLine.serial_number
                        ).in_(payload_serials),
                    )
                    .all()
                )
                if serial
            }

            if existing_line_serials:
                duplicate = sorted(
                    existing_line_serials
                )[0]

                raise ApplianceReceivingError(
                    f"Serial already exists in Receiving: "
                    f"{duplicate}"
                )

            existing_unit_serials = {
                str(serial).strip().upper()
                for (serial,) in (
                    db.session.query(
                        ApplianceUnit.serial_number
                    )
                    .filter(
                        ApplianceUnit.serial_number
                        .isnot(None),
                        func.upper(
                            ApplianceUnit.serial_number
                        ).in_(payload_serials),
                    )
                    .all()
                )
                if serial
            }

            if existing_unit_serials:
                duplicate = sorted(
                    existing_unit_serials
                )[0]

                raise ApplianceReceivingError(
                    f"Serial already exists in stock: "
                    f"{duplicate}"
                )

        # ----------------------------------------------------
        # Determine starting line number once
        # ----------------------------------------------------

        max_line = (
            db.session.query(
                func.coalesce(
                    func.max(
                        ApplianceReceivingLine.line_no
                    ),
                    0,
                )
            )
            .filter(
                ApplianceReceivingLine.receiving_id
                == receiving.id
            )
            .scalar()
            or 0
        )

        next_line_no = int(max_line) + 1

        created_lines = []

        try:
            for offset, data in enumerate(
                normalized_rows
            ):
                line = ApplianceReceivingLine(
                    receiving_id=receiving.id,
                    line_no=next_line_no + offset,
                    category_id=data["category_id"],
                    brand=data["brand"],
                    model_number=data["model_number"],
                    serial_number=data["serial_number"],
                    description=data["description"],
                    size_value=data["size_value"],
                    size_unit=data["size_unit"],
                    color=data["color"],
                    condition=data["condition"],
                    unit_cost=data["unit_cost"],
                    selling_price=data["selling_price"],
                    notes=data["notes"],
                    created_by_id=actor.id,
                    updated_by_id=actor.id,
                )

                db.session.add(line)
                created_lines.append(line)

            receiving.updated_by_id = actor.id
            receiving.updated_at = datetime.utcnow()

            db.session.commit()

            for line in created_lines:
                db.session.refresh(line)

            return created_lines

        except Exception:
            db.session.rollback()
            raise


    @staticmethod
    def update_line(
        *,
        line_id: int,
        actor: User,
        category_id: int,
        brand: str | None = None,
        model_number: str | None = None,
        serial_number: str | None = None,
        description: str | None = None,
        size_value=None,
        size_unit: str | None = None,
        color: str | None = None,
        condition: str = "new",
        unit_cost=None,
        selling_price=None,
        notes: str | None = None,
    ) -> ApplianceReceivingLine:
        actor = ApplianceReceivingService._require_user(actor)

        line = db.session.get(
            ApplianceReceivingLine,
            line_id,
        )

        if line is None:
            raise ApplianceReceivingError(
                "Receiving line not found."
            )

        receiving = line.receiving

        ApplianceReceivingService._require_draft(receiving)

        ApplianceReceivingService._require_permission(
            actor=actor,
            permission_code="appliance.receive",
            warehouse_id=receiving.warehouse_id,
        )

        ApplianceReceivingService._get_active_category(
            category_id
        )

        serial_clean = (
            ApplianceReceivingService._clean_text(
                serial_number,
                upper=True,
                max_length=160,
            )
        )

        ApplianceReceivingService._check_serial_duplicate(
            serial_number=serial_clean,
            exclude_line_id=line.id,
        )

        new_cost = ApplianceReceivingService._money_or_none(
            unit_cost
        )

        new_price = ApplianceReceivingService._money_or_none(
            selling_price
        )

        financial_changed = (
            new_cost != line.unit_cost
            or new_price != line.selling_price
        )

        if financial_changed:
            ApplianceReceivingService._require_permission(
                actor=actor,
                permission_code="appliance.pricing",
                warehouse_id=receiving.warehouse_id,
            )

        line.category_id = category_id

        line.brand = ApplianceReceivingService._clean_text(
            brand,
            upper=True,
            max_length=100,
        )

        line.model_number = (
            ApplianceReceivingService._clean_text(
                model_number,
                upper=True,
                max_length=120,
            )
        )

        line.serial_number = serial_clean

        line.description = (
            ApplianceReceivingService._clean_text(
                description,
                max_length=500,
            )
        )

        line.size_value = (
            ApplianceReceivingService._float_or_none(
                size_value
            )
        )

        line.size_unit = (
            ApplianceReceivingService._clean_text(
                size_unit,
                max_length=20,
            )
        )

        line.color = (
            ApplianceReceivingService._clean_text(
                color,
                upper=True,
                max_length=80,
            )
        )

        line.condition = (
            ApplianceReceivingService._clean_text(
                condition,
                max_length=40,
            )
            or "new"
        ).lower()

        line.unit_cost = new_cost
        line.selling_price = new_price

        line.notes = (
            ApplianceReceivingService._clean_text(notes)
        )

        line.updated_by_id = actor.id
        line.updated_at = datetime.utcnow()

        receiving.updated_by_id = actor.id
        receiving.updated_at = datetime.utcnow()

        try:
            db.session.commit()
            db.session.refresh(line)
            return line

        except Exception:
            db.session.rollback()
            raise

    @staticmethod
    def delete_line(
        *,
        line_id: int,
        actor: User,
    ) -> None:
        actor = ApplianceReceivingService._require_user(actor)

        line = db.session.get(
            ApplianceReceivingLine,
            line_id,
        )

        if line is None:
            raise ApplianceReceivingError(
                "Receiving line not found."
            )

        receiving = line.receiving

        ApplianceReceivingService._require_draft(receiving)

        ApplianceReceivingService._require_permission(
            actor=actor,
            permission_code="appliance.receive",
            warehouse_id=receiving.warehouse_id,
        )

        receiving.updated_by_id = actor.id
        receiving.updated_at = datetime.utcnow()

        try:
            db.session.delete(line)
            db.session.commit()

        except Exception:
            db.session.rollback()
            raise

    @staticmethod
    def delete_draft(
        *,
        receiving_id: int,
        actor: User,
    ) -> None:
        """
        Permanently delete an unfinished Draft Receiving.

        This operation is intentionally limited to Draft documents.
        Posted Receiving must never be hard-deleted through this method.

        Draft lines are removed by the ApplianceReceiving.lines
        delete-orphan cascade.

        As an additional integrity guard, deletion is refused if any
        ApplianceUnit already references a line from this Receiving.
        """
        actor = ApplianceReceivingService._require_user(actor)

        receiving = (
            ApplianceReceivingService._get_receiving(
                receiving_id
            )
        )

        ApplianceReceivingService._require_draft(receiving)

        ApplianceReceivingService._require_permission(
            actor=actor,
            permission_code="appliance.receiving.delete_draft",
            warehouse_id=receiving.warehouse_id,
        )

        unit_exists = (
            ApplianceUnit.query
            .join(
                ApplianceReceivingLine,
                ApplianceUnit.receiving_line_id
                == ApplianceReceivingLine.id,
            )
            .filter(
                ApplianceReceivingLine.receiving_id
                == receiving.id
            )
            .first()
        )

        if unit_exists is not None:
            raise ApplianceReceivingError(
                "Draft Receiving cannot be deleted because "
                "an ApplianceUnit already references one of "
                "its Receiving lines."
            )

        try:
            db.session.delete(receiving)
            db.session.commit()

        except Exception:
            db.session.rollback()
            raise


    # --------------------------------------------------------
    # Posting
    # --------------------------------------------------------

    @staticmethod
    def post_receiving(
        *,
        receiving_id: int,
        actor: User,
    ) -> ApplianceReceiving:
        """
        Finalize receiving and create serialized ApplianceUnit rows.

        IMPORTANT:
        Missing unit_cost does NOT block posting.
        Those rows appear in the Manager Missing Prices queue.
        """

        _authorize_mutation("appliance.receiving.post")
        actor = ApplianceReceivingService._require_user(actor)

        receiving = (
            ApplianceReceivingService._get_receiving(
                receiving_id
            )
        )

        ApplianceReceivingService._require_draft(receiving)

        ApplianceReceivingService._require_permission(
            actor=actor,
            permission_code="appliance.post_receiving",
            warehouse_id=receiving.warehouse_id,
        )

        lines = (
            ApplianceReceivingLine.query
            .filter_by(receiving_id=receiving.id)
            .order_by(
                ApplianceReceivingLine.line_no.asc()
            )
            .all()
        )

        if not lines:
            raise ApplianceReceivingError(
                "Cannot post an empty Receiving."
            )

        try:
            for line in lines:
                existing_unit = (
                    ApplianceUnit.query
                    .filter_by(
                        receiving_line_id=line.id
                    )
                    .first()
                )

                if existing_unit is not None:
                    raise ApplianceReceivingError(
                        f"Receiving line {line.line_no} "
                        f"already created ApplianceUnit "
                        f"{existing_unit.inventory_number}."
                    )

                if line.serial_number:
                    serial_exists = (
                        ApplianceUnit.query
                        .filter(
                            func.upper(
                                ApplianceUnit.serial_number
                            )
                            == line.serial_number.upper()
                        )
                        .first()
                    )

                    if serial_exists is not None:
                        raise ApplianceReceivingError(
                            f"Duplicate serial number: "
                            f"{line.serial_number}"
                        )

                unit = ApplianceUnit(
                    inventory_number=(
                        ApplianceReceivingService
                        ._next_inventory_number()
                    ),
                    category_id=line.category_id,
                    warehouse_id=receiving.warehouse_id,
                    receiving_line_id=line.id,
                    brand=line.brand,
                    model_number=line.model_number,
                    serial_number=line.serial_number,
                    description=line.description,
                    size_value=line.size_value,
                    size_unit=line.size_unit,
                    color=line.color,
                    condition=line.condition,
                    status=(
                        ApplianceReceivingService
                        .UNIT_STATUS_AVAILABLE
                    ),
                    unit_cost=line.unit_cost,
                    selling_price=line.selling_price,
                    notes=line.notes,
                    created_by_id=actor.id,
                    updated_by_id=actor.id,
                )

                db.session.add(unit)

                # Flush is required so the next inventory number
                # sees the ID generated for this unit.
                db.session.flush()

            receiving.status = (
                ApplianceReceivingService.STATUS_POSTED
            )

            receiving.posted_at = datetime.utcnow()
            receiving.posted_by_id = actor.id
            receiving.updated_at = datetime.utcnow()
            receiving.updated_by_id = actor.id

            db.session.commit()
            db.session.refresh(receiving)

            return receiving

        except Exception:
            db.session.rollback()
            raise

    # --------------------------------------------------------
    # Emergency Purge - dependency check
    # --------------------------------------------------------

    @staticmethod
    def check_emergency_purge(
        *,
        receiving_id: int,
        actor: User,
    ) -> dict:
        """
        Read-only dependency analysis for Emergency Purge.

        This method NEVER deletes or modifies data.

        Emergency Purge is a break-glass SUPERADMIN operation.
        It may eventually remove an erroneous Receiving and its
        technical inventory history only when no downstream
        business dependency exists.

        Technical/audit movements allowed during purge analysis:
            DETAILS_UPDATE
            IDENTITY_CORRECTION
            MODEL_SPECS_SYNC
            RECEIVING_VOID

        Any unknown or operational movement blocks purge.
        """
        actor = ApplianceReceivingService._require_user(actor)

        role = (
            getattr(actor, "role", "")
            or ""
        ).strip().lower()

        if role != "superadmin":
            raise ApplianceAccessDenied(
                "Only SUPERADMIN can perform "
                "Emergency Purge analysis."
            )

        receiving = (
            ApplianceReceivingService._get_receiving(
                receiving_id
            )
        )

        lines = (
            ApplianceReceivingLine.query
            .filter_by(
                receiving_id=receiving.id
            )
            .order_by(
                ApplianceReceivingLine.line_no.asc()
            )
            .all()
        )

        units = (
            ApplianceUnit.query
            .join(
                ApplianceReceivingLine,
                ApplianceUnit.receiving_line_id
                == ApplianceReceivingLine.id,
            )
            .filter(
                ApplianceReceivingLine.receiving_id
                == receiving.id
            )
            .order_by(
                ApplianceUnit.id.asc()
            )
            .all()
        )

        technical_movement_types = {
            "DETAILS_UPDATE",
            "IDENTITY_CORRECTION",
            "MODEL_SPECS_SYNC",
            "RECEIVING_VOID",
        }

        blockers = []
        technical_history = []
        unit_results = []

        for unit in units:
            unit_blockers = []
            unit_technical = []

            # ------------------------------------------------
            # Current status guard
            # ------------------------------------------------

            if unit.status not in {
                "available",
                "voided",
            }:
                unit_blockers.append(
                    "Current inventory status is "
                    f"{unit.status!r}."
                )

            # ------------------------------------------------
            # Current Work Order guard
            # ------------------------------------------------

            if (
                unit.current_work_order_id is not None
                or (
                    unit.current_work_order_number
                    and str(
                        unit.current_work_order_number
                    ).strip()
                )
            ):
                wo_ref = (
                    str(
                        unit.current_work_order_number
                    ).strip()
                    if unit.current_work_order_number
                    else str(
                        unit.current_work_order_id
                    )
                )

                unit_blockers.append(
                    "Current Work Order reference exists: "
                    f"{wo_ref}."
                )

            # ------------------------------------------------
            # Issue history is always business history.
            # ------------------------------------------------

            issue_lines = (
                ApplianceIssueLine.query
                .filter_by(
                    appliance_unit_id=unit.id
                )
                .order_by(
                    ApplianceIssueLine.id.asc()
                )
                .all()
            )

            for issue_line in issue_lines:
                unit_blockers.append(
                    "Appliance Issue history exists: "
                    f"IssueLine #{issue_line.id}, "
                    f"Issue #{issue_line.issue_id}, "
                    f"status={issue_line.status!r}."
                )

            # ------------------------------------------------
            # Repair Order history is always business history.
            #
            # Do not rely only on movement history here.
            # ApplianceRepairOrder has its own direct FK to the
            # physical ApplianceUnit and must independently block
            # Emergency Purge.
            # ------------------------------------------------

            from models import ApplianceRepairOrder

            repair_orders = (
                ApplianceRepairOrder.query
                .filter_by(
                    appliance_unit_id=unit.id
                )
                .order_by(
                    ApplianceRepairOrder.id.asc()
                )
                .all()
            )

            for repair_order in repair_orders:
                unit_blockers.append(
                    "Repair Order history exists: "
                    f"{repair_order.repair_number} "
                    f"(RepairOrder #{repair_order.id}, "
                    f"status={repair_order.status!r})."
                )

            # ------------------------------------------------
            # Movements owned by this AP.
            # ------------------------------------------------

            movements = (
                ApplianceMovement.query
                .filter_by(
                    appliance_unit_id=unit.id
                )
                .order_by(
                    ApplianceMovement.created_at.asc(),
                    ApplianceMovement.id.asc(),
                )
                .all()
            )

            for movement in movements:
                movement_type = (
                    movement.movement_type
                    or ""
                ).strip()

                if (
                    movement_type
                    in technical_movement_types
                ):
                    unit_technical.append(
                        movement_type
                    )
                else:
                    unit_blockers.append(
                        "Operational lifecycle movement exists: "
                        f"{movement_type or 'UNKNOWN'} "
                        f"(Movement #{movement.id})."
                    )

            # ------------------------------------------------
            # Another movement may point TO this AP through
            # related_appliance_unit_id (replace/exchange).
            # Never silently erase that historical relation.
            # ------------------------------------------------

            related_movements = (
                ApplianceMovement.query
                .filter(
                    ApplianceMovement
                    .related_appliance_unit_id
                    == unit.id
                )
                .order_by(
                    ApplianceMovement.created_at.asc(),
                    ApplianceMovement.id.asc(),
                )
                .all()
            )

            for movement in related_movements:
                unit_blockers.append(
                    "Another appliance movement references "
                    f"{unit.inventory_number}: "
                    f"{movement.movement_type or 'UNKNOWN'} "
                    f"(Movement #{movement.id}, "
                    f"owner AP id "
                    f"{movement.appliance_unit_id})."
                )

            if unit_technical:
                technical_history.append(
                    {
                        "inventory_number":
                            unit.inventory_number,
                        "movement_types":
                            sorted(
                                set(
                                    unit_technical
                                )
                            ),
                    }
                )

            if unit_blockers:
                blockers.append(
                    {
                        "inventory_number":
                            unit.inventory_number,
                        "reasons":
                            unit_blockers,
                    }
                )

            unit_results.append(
                {
                    "id": unit.id,
                    "inventory_number":
                        unit.inventory_number,
                    "status":
                        unit.status,
                    "safe":
                        not bool(
                            unit_blockers
                        ),
                    "blockers":
                        unit_blockers,
                    "technical_history":
                        sorted(
                            set(
                                unit_technical
                            )
                        ),
                }
            )

        # ----------------------------------------------------
        # Integrity guard:
        #
        # A POSTED / VOIDED Receiving should have exactly one
        # ApplianceUnit per Receiving line.
        # ----------------------------------------------------

        receiving_status = (
            receiving.status
            or ""
        ).strip().lower()

        if receiving_status in {
            "posted",
            "voided",
        }:
            if len(lines) != len(units):
                blockers.append(
                    {
                        "inventory_number":
                            None,
                        "reasons": [
                            (
                                "Receiving inventory integrity "
                                "check failed: "
                                f"{len(lines)} line(s), "
                                f"{len(units)} ApplianceUnit "
                                "record(s)."
                            )
                        ],
                    }
                )

        # Draft Receiving must never already own units.
        if (
            receiving_status == "draft"
            and units
        ):
            blockers.append(
                {
                    "inventory_number":
                        None,
                    "reasons": [
                        (
                            "Draft Receiving unexpectedly "
                            "already has ApplianceUnit records."
                        )
                    ],
                }
            )

        return {
            "safe": not bool(blockers),
            "receiving_id": receiving.id,
            "receiving_number":
                receiving.receiving_number,
            "receiving_status":
                receiving.status,
            "line_count": len(lines),
            "unit_count": len(units),
            "units": unit_results,
            "technical_history":
                technical_history,
            "blockers": blockers,
        }

    # --------------------------------------------------------
    # Emergency Purge
    # --------------------------------------------------------

    @staticmethod
    def emergency_purge(
        *,
        receiving_id: int,
        actor: User,
        confirmation: str,
    ) -> dict:
        """
        Physically remove an erroneous POSTED / VOIDED Receiving.

        BREAK-GLASS operation:
            - SUPERADMIN only
            - exact confirmation phrase required
            - dependency checker must report SAFE
            - only technical/audit movements may be removed
            - downstream business history always blocks purge

        This is intentionally NOT controlled by the normal
        Permission catalog.
        """
        actor = ApplianceReceivingService._require_user(actor)

        role = (
            getattr(actor, "role", "")
            or ""
        ).strip().lower()

        if role != "superadmin":
            raise ApplianceAccessDenied(
                "Only SUPERADMIN can perform Emergency Purge."
            )

        receiving = (
            ApplianceReceivingService._get_receiving(
                receiving_id
            )
        )

        receiving_status = (
            receiving.status
            or ""
        ).strip().lower()

        if receiving_status not in {
            "posted",
            "voided",
        }:
            raise ApplianceReceivingError(
                "Emergency Purge is allowed only for "
                "POSTED or VOIDED Receiving. "
                "Use normal Delete Draft for DRAFT Receiving."
            )

        receiving_number = (
            receiving.receiving_number
            or ""
        ).strip()

        expected_confirmation = (
            f"PURGE {receiving_number}"
        )

        actual_confirmation = (
            confirmation
            or ""
        ).strip()

        if actual_confirmation != expected_confirmation:
            raise ApplianceReceivingError(
                "Emergency Purge confirmation does not match. "
                f'Type exactly: "{expected_confirmation}"'
            )

        # ----------------------------------------------------
        # Full dependency analysis immediately before purge.
        # ----------------------------------------------------

        inspection = (
            ApplianceReceivingService
            .check_emergency_purge(
                receiving_id=receiving.id,
                actor=actor,
            )
        )

        if not inspection.get("safe"):
            raise ApplianceReceivingError(
                "Emergency Purge blocked because downstream "
                "dependencies exist."
            )

        technical_movement_types = {
            "DETAILS_UPDATE",
            "IDENTITY_CORRECTION",
            "MODEL_SPECS_SYNC",
            "RECEIVING_VOID",
        }

        try:
            # ------------------------------------------------
            # Reload Receiving and all physical units.
            # ------------------------------------------------

            receiving = (
                ApplianceReceivingService._get_receiving(
                    receiving_id
                )
            )

            units = (
                ApplianceUnit.query
                .join(
                    ApplianceReceivingLine,
                    ApplianceUnit.receiving_line_id
                    == ApplianceReceivingLine.id,
                )
                .filter(
                    ApplianceReceivingLine.receiving_id
                    == receiving.id
                )
                .order_by(
                    ApplianceUnit.id.asc()
                )
                .all()
            )

            unit_ids = [
                unit.id
                for unit in units
            ]

            deleted_movements = 0
            deleted_units = 0

            # ------------------------------------------------
            # Final fail-safe movement validation.
            #
            # Do not trust only the earlier inspection result.
            # Re-read immediately before physical deletion.
            # ------------------------------------------------

            if unit_ids:
                movements = (
                    ApplianceMovement.query
                    .filter(
                        ApplianceMovement.appliance_unit_id.in_(
                            unit_ids
                        )
                    )
                    .order_by(
                        ApplianceMovement.id.asc()
                    )
                    .all()
                )

                for movement in movements:
                    movement_type = (
                        movement.movement_type
                        or ""
                    ).strip()

                    if (
                        movement_type
                        not in technical_movement_types
                    ):
                        raise ApplianceReceivingError(
                            "Emergency Purge aborted: "
                            "operational or unknown movement "
                            "appeared before deletion: "
                            f"{movement_type or 'UNKNOWN'} "
                            f"(Movement #{movement.id})."
                        )

                # --------------------------------------------
                # A movement belonging to another AP must not
                # reference any AP being purged.
                # --------------------------------------------

                external_related = (
                    ApplianceMovement.query
                    .filter(
                        ApplianceMovement
                        .related_appliance_unit_id
                        .in_(unit_ids),
                        ~ApplianceMovement
                        .appliance_unit_id
                        .in_(unit_ids),
                    )
                    .order_by(
                        ApplianceMovement.id.asc()
                    )
                    .first()
                )

                if external_related is not None:
                    raise ApplianceReceivingError(
                        "Emergency Purge aborted: "
                        "another appliance movement references "
                        "an AP from this Receiving "
                        f"(Movement #{external_related.id})."
                    )

                # --------------------------------------------
                # Remove technical movements first because
                # appliance_movement.appliance_unit_id uses
                # ON DELETE RESTRICT.
                # --------------------------------------------

                for movement in movements:
                    db.session.delete(
                        movement
                    )
                    deleted_movements += 1

                db.session.flush()

                # --------------------------------------------
                # Remove physical inventory units.
                # --------------------------------------------

                for unit in units:
                    db.session.delete(
                        unit
                    )
                    deleted_units += 1

                db.session.flush()

            # ------------------------------------------------
            # Removing Receiving removes its ReceivingLine
            # rows through the existing ORM delete-orphan
            # relationship.
            # ------------------------------------------------

            line_count = (
                ApplianceReceivingLine.query
                .filter_by(
                    receiving_id=receiving.id
                )
                .count()
            )

            purged_receiving_id = receiving.id
            purged_receiving_number = (
                receiving.receiving_number
            )

            db.session.delete(
                receiving
            )

            db.session.commit()

            return {
                "purged": True,
                "receiving_id":
                    purged_receiving_id,
                "receiving_number":
                    purged_receiving_number,
                "deleted_lines":
                    line_count,
                "deleted_units":
                    deleted_units,
                "deleted_movements":
                    deleted_movements,
            }

        except Exception:
            db.session.rollback()
            raise


    # --------------------------------------------------------
    # Void Posted Receiving
    # --------------------------------------------------------

    @staticmethod
    def void_receiving(
        *,
        receiving_id: int,
        actor: User,
        reason: str,
    ) -> ApplianceReceiving:
        """
        Formally reverse a Posted Appliance Receiving.

        VOID is intentionally conservative.

        It is allowed only when every ApplianceUnit created by this
        Receiving is still in its untouched post-receiving state:

            - Receiving status is POSTED
            - one ApplianceUnit exists for every Receiving line
            - every ApplianceUnit status is AVAILABLE
            - no ApplianceUnit is attached to a current Work Order
            - no ApplianceMovement exists for any ApplianceUnit

        Nothing is physically deleted.

        Each ApplianceUnit is marked VOIDED and receives an immutable
        RECEIVING_VOID movement. The Receiving itself is marked VOIDED
        with actor, timestamp, and reason.

        All changes commit atomically.
        """

        _authorize_mutation("appliance.receiving.void")
        actor = ApplianceReceivingService._require_user(actor)

        receiving = (
            ApplianceReceivingService._get_receiving(
                receiving_id
            )
        )

        if receiving.status != (
            ApplianceReceivingService.STATUS_POSTED
        ):
            raise ApplianceReceivingError(
                "Only a Posted Receiving can be voided."
            )

        ApplianceReceivingService._require_permission(
            actor=actor,
            permission_code="appliance.receiving.void",
            warehouse_id=receiving.warehouse_id,
        )

        reason_clean = ApplianceReceivingService._clean_text(
            reason,
            max_length=1000,
        )

        if reason_clean is None:
            raise ApplianceReceivingError(
                "VOID reason is required."
            )

        lines = (
            ApplianceReceivingLine.query
            .filter_by(receiving_id=receiving.id)
            .order_by(
                ApplianceReceivingLine.line_no.asc()
            )
            .all()
        )

        if not lines:
            raise ApplianceReceivingError(
                "Posted Receiving has no Receiving lines. "
                "VOID refused."
            )

        units = (
            ApplianceUnit.query
            .join(
                ApplianceReceivingLine,
                ApplianceUnit.receiving_line_id
                == ApplianceReceivingLine.id,
            )
            .filter(
                ApplianceReceivingLine.receiving_id
                == receiving.id
            )
            .order_by(
                ApplianceUnit.id.asc()
            )
            .all()
        )

        if len(units) != len(lines):
            raise ApplianceReceivingError(
                "Receiving inventory integrity check failed: "
                f"{len(lines)} Receiving line(s), but "
                f"{len(units)} ApplianceUnit record(s) found. "
                "VOID refused."
            )

        # ----------------------------------------------------
        # Validate ALL units before changing anything.
        # ----------------------------------------------------
        for unit in units:
            if unit.status != (
                ApplianceReceivingService
                .UNIT_STATUS_AVAILABLE
            ):
                raise ApplianceReceivingError(
                    f"{unit.inventory_number} cannot be voided: "
                    f"current status is {unit.status!r}."
                )

            if (
                unit.current_work_order_id is not None
                or (
                    unit.current_work_order_number
                    and str(
                        unit.current_work_order_number
                    ).strip()
                )
            ):
                raise ApplianceReceivingError(
                    f"{unit.inventory_number} cannot be voided: "
                    "it is or was left attached to a current "
                    "Work Order."
                )

            # Audit/correction movements do not represent physical
            # lifecycle usage and therefore do not block Receiving VOID.
            #
            # Everything else is treated as lifecycle history.
            # This is intentionally fail-safe: any future operational
            # movement type will automatically block VOID unless it is
            # explicitly classified as audit-only here.
            audit_only_movement_types = {
                "DETAILS_UPDATE",
                "IDENTITY_CORRECTION",
                "MODEL_SPECS_SYNC",
            }

            lifecycle_movement = (
                ApplianceMovement.query
                .filter(
                    ApplianceMovement.appliance_unit_id
                    == unit.id,
                    ~ApplianceMovement.movement_type.in_(
                        audit_only_movement_types
                    ),
                )
                .order_by(
                    ApplianceMovement.created_at.asc(),
                    ApplianceMovement.id.asc(),
                )
                .first()
            )

            if lifecycle_movement is not None:
                raise ApplianceReceivingError(
                    f"{unit.inventory_number} cannot be voided: "
                    "lifecycle history already exists "
                    f"({lifecycle_movement.movement_type})."
                )

        # ----------------------------------------------------
        # Validation passed for the whole Receiving.
        # Apply reversal atomically.
        # ----------------------------------------------------
        now = datetime.utcnow()

        try:
            for unit in units:
                unit.status = (
                    ApplianceReceivingService
                    .UNIT_STATUS_VOIDED
                )
                unit.updated_at = now
                unit.updated_by_id = actor.id

                movement = ApplianceMovement(
                    appliance_unit_id=unit.id,
                    movement_type="RECEIVING_VOID",
                    from_warehouse_id=unit.warehouse_id,
                    to_warehouse_id=None,
                    reason_code="RECEIVING_VOID",
                    notes=reason_clean,
                    actor_id=actor.id,
                    created_at=now,
                )

                db.session.add(movement)

            receiving.status = (
                ApplianceReceivingService.STATUS_VOIDED
            )
            receiving.voided_at = now
            receiving.voided_by_id = actor.id
            receiving.void_reason = reason_clean
            receiving.updated_at = now
            receiving.updated_by_id = actor.id

            db.session.commit()
            db.session.refresh(receiving)

            return receiving

        except Exception:
            db.session.rollback()
            raise

    # --------------------------------------------------------
    # Posted Receiving - descriptive fields / pricing
    # --------------------------------------------------------

    @staticmethod
    def update_posted_line_fields(
        *,
        line_id: int,
        actor: User,
        update_details: bool = False,
        update_pricing: bool = False,
        size_value=None,
        size_unit=None,
        color=None,
        notes=None,
        unit_cost=None,
        selling_price=None,
    ) -> ApplianceReceivingLine:
        """
        Safely update allowed fields after Receiving is posted.

        Warehouse users with appliance.receive may update:
            - size_value
            - size_unit
            - color
            - notes

        Users with appliance.pricing may update:
            - unit_cost
            - selling_price

        Receiving history and current ApplianceUnit are kept
        synchronized in the same transaction.
        """
        actor = ApplianceReceivingService._require_user(actor)

        line = db.session.get(
            ApplianceReceivingLine,
            line_id,
        )

        if line is None:
            raise ApplianceReceivingError(
                "Receiving line not found."
            )

        receiving = line.receiving

        if receiving.status != (
            ApplianceReceivingService.STATUS_POSTED
        ):
            raise ApplianceReceivingError(
                "This action is available only after "
                "Receiving is posted."
            )

        if not update_details and not update_pricing:
            raise ApplianceReceivingError(
                "No editable fields were submitted."
            )

        # ----------------------------------------------------
        # Permission checks are independent.
        # Warehouse descriptive editing must NOT grant pricing.
        # ----------------------------------------------------

        if update_details:
            ApplianceReceivingService._require_permission(
                actor=actor,
                permission_code="appliance.receive",
                warehouse_id=receiving.warehouse_id,
            )

        if update_pricing:
            ApplianceReceivingService._require_permission(
                actor=actor,
                permission_code="appliance.pricing",
                warehouse_id=receiving.warehouse_id,
            )

        # ----------------------------------------------------
        # Find physical appliance created from this line.
        # ----------------------------------------------------

        unit = (
            ApplianceUnit.query
            .filter_by(
                receiving_line_id=line.id
            )
            .first()
        )

        if unit is None:
            raise ApplianceReceivingError(
                "ApplianceUnit for this Receiving line "
                "was not found."
            )

        # ----------------------------------------------------
        # Normalize descriptive fields.
        # ----------------------------------------------------

        new_size_value = None
        new_size_unit = None
        new_color = None
        new_notes = None

        if update_details:

            raw_size = (
                str(size_value).strip()
                if size_value is not None
                else ""
            )

            if raw_size:
                try:
                    new_size_value = float(raw_size)
                except (TypeError, ValueError):
                    raise ApplianceReceivingError(
                        "Size must be a valid number."
                    )

                if new_size_value < 0:
                    raise ApplianceReceivingError(
                        "Size cannot be negative."
                    )

            raw_unit = (
                str(size_unit or "")
                .strip()
                .upper()
            )

            if len(raw_unit) > 20:
                raise ApplianceReceivingError(
                    "Size Unit cannot exceed 20 characters."
                )

            new_size_unit = raw_unit or None

            raw_color = (
                str(color or "")
                .strip()
                .upper()
            )

            if len(raw_color) > 80:
                raise ApplianceReceivingError(
                    "Color cannot exceed 80 characters."
                )

            new_color = raw_color or None

            raw_notes = (
                str(notes or "")
                .strip()
            )

            if len(raw_notes) > 5000:
                raise ApplianceReceivingError(
                    "Notes cannot exceed 5000 characters."
                )

            new_notes = raw_notes or None

        # ----------------------------------------------------
        # Normalize pricing only when authorized.
        # ----------------------------------------------------

        new_cost = None
        new_price = None

        if update_pricing:
            new_cost = (
                ApplianceReceivingService._money_or_none(
                    unit_cost
                )
            )

            new_price = (
                ApplianceReceivingService._money_or_none(
                    selling_price
                )
            )

        now = datetime.utcnow()

        try:

            # ------------------------------------------------
            # Descriptive data
            # ------------------------------------------------

            if update_details:

                line.size_value = new_size_value
                line.size_unit = new_size_unit
                line.color = new_color
                line.notes = new_notes

                unit.size_value = new_size_value
                unit.size_unit = new_size_unit
                unit.color = new_color
                unit.notes = new_notes

            # ------------------------------------------------
            # Pricing
            # ------------------------------------------------

            if update_pricing:

                line.unit_cost = new_cost
                line.selling_price = new_price

                unit.unit_cost = new_cost
                unit.selling_price = new_price

            # ------------------------------------------------
            # Audit timestamps / actor
            # ------------------------------------------------

            line.updated_at = now
            line.updated_by_id = actor.id

            unit.updated_at = now
            unit.updated_by_id = actor.id

            receiving.updated_at = now
            receiving.updated_by_id = actor.id

            db.session.commit()
            db.session.refresh(line)

            return line

        except Exception:
            db.session.rollback()
            raise


    # --------------------------------------------------------
    # Manager Pricing
    # --------------------------------------------------------

    @staticmethod
    def update_posted_line_price(
        *,
        line_id: int,
        actor: User,
        unit_cost=None,
        selling_price=None,
    ) -> ApplianceReceivingLine:
        """
        Manager may fill/correct prices after Receiving is posted.

        Receiving history and current ApplianceUnit are updated together
        in one transaction.
        """
        actor = ApplianceReceivingService._require_user(actor)

        line = db.session.get(
            ApplianceReceivingLine,
            line_id,
        )

        if line is None:
            raise ApplianceReceivingError(
                "Receiving line not found."
            )

        receiving = line.receiving

        if receiving.status != (
            ApplianceReceivingService.STATUS_POSTED
        ):
            raise ApplianceReceivingError(
                "Use Draft editing before Receiving is posted."
            )

        ApplianceReceivingService._require_permission(
            actor=actor,
            permission_code="appliance.pricing",
            warehouse_id=receiving.warehouse_id,
        )

        new_cost = ApplianceReceivingService._money_or_none(
            unit_cost
        )

        new_price = ApplianceReceivingService._money_or_none(
            selling_price
        )

        unit = (
            ApplianceUnit.query
            .filter_by(
                receiving_line_id=line.id
            )
            .first()
        )

        if unit is None:
            raise ApplianceReceivingError(
                "ApplianceUnit for this Receiving line "
                "was not found."
            )

        try:
            line.unit_cost = new_cost
            line.selling_price = new_price
            line.updated_at = datetime.utcnow()
            line.updated_by_id = actor.id

            unit.unit_cost = new_cost
            unit.selling_price = new_price
            unit.updated_at = datetime.utcnow()
            unit.updated_by_id = actor.id

            receiving.updated_at = datetime.utcnow()
            receiving.updated_by_id = actor.id

            db.session.commit()
            db.session.refresh(line)

            return line

        except Exception:
            db.session.rollback()
            raise
