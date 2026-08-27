from __future__ import annotations

from datetime import datetime

from flask import (
    flash,
    redirect,
    render_template,
    request,
    url_for,
    jsonify,
)
from flask_login import current_user, login_required
from sqlalchemy import or_

from extensions import db
from models import (
    ApplianceCategory,
    ApplianceReceiving,
    ApplianceReceivingLine,
    ApplianceUnit,
)
from appliance import appliance_bp
from services.access_control_service import AccessControlService
from services.appliance_receiving_service import (
    ApplianceAccessDenied,
    ApplianceReceivingError,
    ApplianceReceivingService,
)


def _parse_date(value: str | None):
    value = (value or "").strip()

    if not value:
        return None

    try:
        return datetime.strptime(
            value,
            "%Y-%m-%d",
        ).date()
    except ValueError:
        raise ApplianceReceivingError(
            "Invalid date format."
        )


def _active_categories():
    return (
        ApplianceCategory.query
        .filter(ApplianceCategory.is_active.is_(True))
        .order_by(
            ApplianceCategory.sort_order.asc(),
            ApplianceCategory.name.asc(),
        )
        .all()
    )


def _warehouse_ids_for(permission_code: str):
    return [
        warehouse.id
        for warehouse in AccessControlService.accessible_warehouses(
            current_user
        )
        if AccessControlService.can(
            current_user,
            permission_code,
            warehouse_id=warehouse.id,
        )
    ]


def _form_line_kwargs():
    return {
        "category_id": int(
            request.form.get("category_id") or 0
        ),
        "brand": request.form.get("brand"),
        "model_number": request.form.get("model_number"),
        "serial_number": request.form.get("serial_number"),
        "description": request.form.get("description"),
        "size_value": request.form.get("size_value"),
        "size_unit": request.form.get("size_unit"),
        "condition": (
            request.form.get("condition")
            or "new"
        ),
        "unit_cost": request.form.get("unit_cost"),
        "selling_price": request.form.get("selling_price"),
        "notes": request.form.get("notes"),
    }


# ============================================================
# Receiving list
# ============================================================

@appliance_bp.get("/receiving")
@login_required
def receiving_list():
    allowed_ids = _warehouse_ids_for(
        "appliance.view"
    )

    q = (request.args.get("q") or "").strip()
    status = (
        request.args.get("status") or ""
    ).strip().lower()

    if not allowed_ids:
        receivings = []
    else:
        query = (
            ApplianceReceiving.query
            .filter(
                ApplianceReceiving.warehouse_id.in_(
                    allowed_ids
                )
            )
        )

        if status in {"draft", "posted"}:
            query = query.filter(
                ApplianceReceiving.status == status
            )

        if q:
            like = f"%{q}%"

            query = query.filter(
                or_(
                    ApplianceReceiving.receiving_number.ilike(
                        like
                    ),
                    ApplianceReceiving.supplier_name.ilike(
                        like
                    ),
                    ApplianceReceiving.invoice_number.ilike(
                        like
                    ),
                )
            )

        receivings = (
            query
            .order_by(
                ApplianceReceiving.received_at.desc(),
                ApplianceReceiving.id.desc(),
            )
            .limit(300)
            .all()
        )

    missing_ids = set()

    pricing_warehouse_ids = _warehouse_ids_for(
        "appliance.pricing"
    )

    if pricing_warehouse_ids:
        missing_ids = {
            row.id
            for row in (
                ApplianceReceiving.query
                .join(
                    ApplianceReceivingLine,
                    ApplianceReceivingLine.receiving_id
                    == ApplianceReceiving.id,
                )
                .filter(
                    ApplianceReceiving.warehouse_id.in_(
                        pricing_warehouse_ids
                    ),
                    ApplianceReceiving.status == "posted",
                    ApplianceReceivingLine.unit_cost.is_(None),
                )
                .distinct()
                .all()
            )
        }

    can_receive = bool(
        _warehouse_ids_for(
            "appliance.receive"
        )
    )

    return render_template(
        "appliance_receiving_list.html",
        receivings=receivings,
        missing_ids=missing_ids,
        missing_count=len(missing_ids),
        q=q,
        status=status,
        can_receive=can_receive,
        missing_only=False,
    )


# ============================================================
# Missing Prices queue
# ============================================================

@appliance_bp.get("/receiving/missing-prices")
@login_required
def receiving_missing_prices():
    receivings = (
        ApplianceReceivingService
        .list_missing_price_receivings(
            actor=current_user,
        )
    )

    missing_ids = {
        row.id
        for row in receivings
    }

    return render_template(
        "appliance_receiving_list.html",
        receivings=receivings,
        missing_ids=missing_ids,
        missing_count=len(missing_ids),
        q="",
        status="posted",
        can_receive=bool(
            _warehouse_ids_for(
                "appliance.receive"
            )
        ),
        missing_only=True,
    )


# ============================================================
# New Receiving
# ============================================================

@appliance_bp.route(
    "/receiving/new",
    methods=["GET", "POST"],
)
@login_required
def receiving_new():
    warehouses = [
        warehouse
        for warehouse in AccessControlService.accessible_warehouses(
            current_user
        )
        if AccessControlService.can(
            current_user,
            "appliance.receive",
            warehouse_id=warehouse.id,
        )
    ]

    if not warehouses:
        flash(
            "You do not have Appliance Receiving access.",
            "danger",
        )
        return redirect(
            url_for("appliance.receiving_list")
        )

    if request.method == "POST":
        try:
            warehouse_id = int(
                request.form.get("warehouse_id") or 0
            )

            receiving = (
                ApplianceReceivingService.create_draft(
                    actor=current_user,
                    warehouse_id=warehouse_id,
                    supplier_name=request.form.get(
                        "supplier_name"
                    ),
                    invoice_number=request.form.get(
                        "invoice_number"
                    ),
                    invoice_date=_parse_date(
                        request.form.get(
                            "invoice_date"
                        )
                    ),
                    notes=request.form.get("notes"),
                )
            )

            flash(
                f"Receiving "
                f"{receiving.receiving_number} created.",
                "success",
            )

            return redirect(
                url_for(
                    "appliance.receiving_detail",
                    receiving_id=receiving.id,
                )
            )

        except (
            ApplianceReceivingError,
            ValueError,
        ) as exc:
            db.session.rollback()
            flash(str(exc), "danger")

    return render_template(
        "appliance_receiving_form.html",
        receiving=None,
        warehouses=warehouses,
        categories=_active_categories(),
        can_receive=True,
        can_post=False,
        can_pricing=False,
    )


# ============================================================
# Receiving detail / edit
# ============================================================

@appliance_bp.get(
    "/receiving/<int:receiving_id>"
)
@login_required
def receiving_detail(receiving_id):
    try:
        receiving = (
            ApplianceReceivingService.get_receiving(
                receiving_id=receiving_id,
                actor=current_user,
                permission_code="appliance.view",
            )
        )

    except ApplianceReceivingError as exc:
        flash(str(exc), "danger")
        return redirect(
            url_for("appliance.receiving_list")
        )

    warehouse_id = receiving.warehouse_id

    can_receive = AccessControlService.can(
        current_user,
        "appliance.receive",
        warehouse_id=warehouse_id,
    )

    can_post = AccessControlService.can(
        current_user,
        "appliance.post_receiving",
        warehouse_id=warehouse_id,
    )

    can_pricing = AccessControlService.can(
        current_user,
        "appliance.pricing",
        warehouse_id=warehouse_id,
    )

    warehouses = (
        AccessControlService.accessible_warehouses(
            current_user
        )
    )

    return render_template(
        "appliance_receiving_form.html",
        receiving=receiving,
        warehouses=warehouses,
        categories=_active_categories(),
        can_receive=can_receive,
        can_post=can_post,
        can_pricing=can_pricing,
    )


# ============================================================
# Draft header
# ============================================================

@appliance_bp.post(
    "/receiving/<int:receiving_id>/header"
)
@login_required
def receiving_update_header(receiving_id):
    try:
        ApplianceReceivingService.update_draft_header(
            receiving_id=receiving_id,
            actor=current_user,
            supplier_name=request.form.get(
                "supplier_name"
            ),
            invoice_number=request.form.get(
                "invoice_number"
            ),
            invoice_date=_parse_date(
                request.form.get("invoice_date")
            ),
            notes=request.form.get("notes"),
        )

        flash(
            "Receiving header saved.",
            "success",
        )

    except ApplianceReceivingError as exc:
        db.session.rollback()
        flash(str(exc), "danger")

    return redirect(
        url_for(
            "appliance.receiving_detail",
            receiving_id=receiving_id,
        )
    )


# ============================================================
# Add physical appliance
# ============================================================

@appliance_bp.post(
    "/receiving/<int:receiving_id>/lines/add"
)
@login_required
def receiving_add_line(receiving_id):
    try:
        kwargs = _form_line_kwargs()

        if not kwargs["category_id"]:
            raise ApplianceReceivingError(
                "Appliance category is required."
            )

        line = ApplianceReceivingService.add_line(
            receiving_id=receiving_id,
            actor=current_user,
            **kwargs,
        )

        flash(
            f"Unit line #{line.line_no} added.",
            "success",
        )

    except (
        ApplianceReceivingError,
        ValueError,
    ) as exc:
        db.session.rollback()
        flash(str(exc), "danger")

    return redirect(
        url_for(
            "appliance.receiving_detail",
            receiving_id=receiving_id,
        )
    )


# ============================================================
# Receiving autocomplete
#
# Historical Brand / Model suggestions from ApplianceUnit.
#
# IMPORTANT:
# - not limited to AVAILABLE inventory;
# - new Brand / Model values are still allowed;
# - suggestions respect category;
# - model suggestions respect category + brand.
# ============================================================

@appliance_bp.get(
    "/receiving/autocomplete"
)
@login_required
def receiving_autocomplete():

    from sqlalchemy import func

    try:
        category_id = int(
            request.args.get("category_id")
            or 0
        )
    except (TypeError, ValueError):
        category_id = 0

    brand = (
        request.args.get("brand")
        or ""
    ).strip().upper()

    if category_id <= 0:
        return jsonify(
            {
                "ok": True,
                "brands": [],
                "models": [],
            }
        )

    # --------------------------------------------------------
    # Access
    # --------------------------------------------------------

    allowed_warehouse_ids = _warehouse_ids_for(
        "appliance.receive"
    )

    if not allowed_warehouse_ids:
        return jsonify(
            {
                "ok": False,
                "error": "Access denied.",
            }
        ), 403

    # --------------------------------------------------------
    # BRAND suggestions
    #
    # We intentionally use the complete ApplianceUnit registry,
    # including issued/sold/etc. units.
    # This behaves as historical warehouse catalog.
    # --------------------------------------------------------

    brand_rows = (
        db.session.query(
            ApplianceUnit.brand
        )
        .filter(
            ApplianceUnit.warehouse_id.in_(
                allowed_warehouse_ids
            ),
            ApplianceUnit.category_id
            == category_id,
            ApplianceUnit.brand.isnot(None),
            func.trim(
                ApplianceUnit.brand
            ) != "",
        )
        .distinct()
        .order_by(
            ApplianceUnit.brand.asc()
        )
        .all()
    )

    brands = sorted(
        {
            (
                row[0]
                or ""
            ).strip().upper()
            for row in brand_rows
            if (
                row[0]
                or ""
            ).strip()
        }
    )

    # --------------------------------------------------------
    # MODEL suggestions
    # --------------------------------------------------------

    models = []

    if brand:

        model_rows = (
            db.session.query(
                ApplianceUnit.model_number
            )
            .filter(
                ApplianceUnit.warehouse_id.in_(
                    allowed_warehouse_ids
                ),
                ApplianceUnit.category_id
                == category_id,
                func.upper(
                    func.trim(
                        ApplianceUnit.brand
                    )
                )
                == brand,
                ApplianceUnit.model_number.isnot(
                    None
                ),
                func.trim(
                    ApplianceUnit.model_number
                ) != "",
            )
            .distinct()
            .order_by(
                ApplianceUnit.model_number.asc()
            )
            .all()
        )

        models = sorted(
            {
                (
                    row[0]
                    or ""
                ).strip().upper()
                for row in model_rows
                if (
                    row[0]
                    or ""
                ).strip()
            }
        )

    return jsonify(
        {
            "ok": True,
            "brands": brands,
            "models": models,
        }
    )


# ============================================================
# Receiving Serial duplicate check
#
# Used by barcode scanner / Batch Entry before moving to
# the next Serial row.
# ============================================================

@appliance_bp.get(
    "/receiving/check-serial"
)
@login_required
def receiving_check_serial():

    serial = (
        request.args.get("serial")
        or ""
    ).strip().upper()

    try:
        receiving_id = int(
            request.args.get("receiving_id")
            or 0
        )
    except (TypeError, ValueError):
        receiving_id = 0

    if not serial:

        return jsonify(
            {
                "ok": True,
                "exists": False,
            }
        )

    # --------------------------------------------------------
    # Verify current Receiving + warehouse access
    # --------------------------------------------------------

    receiving = db.session.get(
        ApplianceReceiving,
        receiving_id,
    )

    if receiving is None:

        return jsonify(
            {
                "ok": False,
                "error": "Receiving not found.",
            }
        ), 404

    if not AccessControlService.can(
        current_user,
        "appliance.receive",
        warehouse_id=receiving.warehouse_id,
    ):

        return jsonify(
            {
                "ok": False,
                "error": "Access denied.",
            }
        ), 403

    # --------------------------------------------------------
    # 1. Already exists as physical ApplianceUnit
    # --------------------------------------------------------

    existing_unit = (
        ApplianceUnit.query
        .filter(
            db.func.upper(
                db.func.trim(
                    ApplianceUnit.serial_number
                )
            )
            == serial
        )
        .first()
    )

    if existing_unit is not None:

        return jsonify(
            {
                "ok": True,
                "exists": True,
                "source": "inventory",
                "message": (
                    f"SERIAL {serial} already exists "
                    f"in Inventory as "
                    f"{existing_unit.inventory_number}."
                ),
                "inventory_number":
                    existing_unit.inventory_number,
            }
        )

    # --------------------------------------------------------
    # 2. Already exists in a saved Receiving line
    #
    # This also catches a Serial entered in another Draft
    # Receiving before it becomes an ApplianceUnit.
    # --------------------------------------------------------

    existing_line = (
        ApplianceReceivingLine.query
        .filter(
            db.func.upper(
                db.func.trim(
                    ApplianceReceivingLine.serial_number
                )
            )
            == serial
        )
        .first()
    )

    if existing_line is not None:

        source_receiving = db.session.get(
            ApplianceReceiving,
            existing_line.receiving_id,
        )

        receiving_number = (
            source_receiving.receiving_number
            if source_receiving is not None
            else ""
        )

        receiving_status = (
            source_receiving.status
            if source_receiving is not None
            else ""
        )

        return jsonify(
            {
                "ok": True,
                "exists": True,
                "source": "receiving",
                "message": (
                    f"SERIAL {serial} already exists "
                    f"in Receiving "
                    f"{receiving_number or '#'+str(existing_line.receiving_id)}"
                    f"{' (' + receiving_status.upper() + ')' if receiving_status else ''}."
                ),
                "receiving_id":
                    existing_line.receiving_id,
                "receiving_number":
                    receiving_number,
            }
        )

    return jsonify(
        {
            "ok": True,
            "exists": False,
        }
    )


# ============================================================
# Bulk Add physical appliances
# ============================================================

@appliance_bp.post(
    "/receiving/<int:receiving_id>/lines/bulk"
)
@login_required
def receiving_add_lines_bulk(receiving_id):
    """
    Excel-like Batch Entry endpoint.

    Entire payload is validated and committed atomically.
    """

    try:
        payload = request.get_json(
            silent=True
        )

        if not isinstance(payload, dict):
            raise ApplianceReceivingError(
                "Invalid request payload."
            )

        rows = payload.get("rows")

        created = (
            ApplianceReceivingService.add_lines_bulk(
                receiving_id=receiving_id,
                actor=current_user,
                rows=rows,
            )
        )

        return jsonify(
            {
                "ok": True,
                "created": len(created),
                "line_ids": [
                    line.id
                    for line in created
                ],
            }
        )

    except ApplianceAccessDenied as exc:
        db.session.rollback()

        return jsonify(
            {
                "ok": False,
                "error": str(exc),
            }
        ), 403

    except (
        ApplianceReceivingError,
        ValueError,
        TypeError,
    ) as exc:
        db.session.rollback()

        return jsonify(
            {
                "ok": False,
                "error": str(exc),
            }
        ), 400

    except Exception:
        db.session.rollback()
        raise


# ============================================================
# Update Draft line
# ============================================================

@appliance_bp.post(
    "/receiving/lines/<int:line_id>/update"
)
@login_required
def receiving_update_line(line_id):
    line = db.session.get(
        ApplianceReceivingLine,
        line_id,
    )

    if line is None:
        flash(
            "Receiving line not found.",
            "danger",
        )
        return redirect(
            url_for("appliance.receiving_list")
        )

    receiving_id = line.receiving_id

    try:
        kwargs = _form_line_kwargs()

        if not kwargs["category_id"]:
            raise ApplianceReceivingError(
                "Appliance category is required."
            )

        ApplianceReceivingService.update_line(
            line_id=line_id,
            actor=current_user,
            **kwargs,
        )

        flash(
            f"Line #{line.line_no} saved.",
            "success",
        )

    except (
        ApplianceReceivingError,
        ValueError,
    ) as exc:
        db.session.rollback()
        flash(str(exc), "danger")

    return redirect(
        url_for(
            "appliance.receiving_detail",
            receiving_id=receiving_id,
        )
    )


# ============================================================
# Delete Draft line
# ============================================================

@appliance_bp.post(
    "/receiving/lines/<int:line_id>/delete"
)
@login_required
def receiving_delete_line(line_id):
    line = db.session.get(
        ApplianceReceivingLine,
        line_id,
    )

    if line is None:
        flash(
            "Receiving line not found.",
            "danger",
        )
        return redirect(
            url_for("appliance.receiving_list")
        )

    receiving_id = line.receiving_id

    try:
        ApplianceReceivingService.delete_line(
            line_id=line_id,
            actor=current_user,
        )

        flash(
            "Appliance line deleted.",
            "success",
        )

    except ApplianceReceivingError as exc:
        db.session.rollback()
        flash(str(exc), "danger")

    return redirect(
        url_for(
            "appliance.receiving_detail",
            receiving_id=receiving_id,
        )
    )


# ============================================================
# Post Receiving
# ============================================================

@appliance_bp.post(
    "/receiving/<int:receiving_id>/post"
)
@login_required
def receiving_post(receiving_id):
    try:
        receiving = (
            ApplianceReceivingService.post_receiving(
                receiving_id=receiving_id,
                actor=current_user,
            )
        )

        if receiving.missing_price_count:
            flash(
                f"{receiving.receiving_number} posted. "
                f"{receiving.missing_price_count} "
                f"unit(s) still need pricing.",
                "warning",
            )
        else:
            flash(
                f"{receiving.receiving_number} posted.",
                "success",
            )

    except ApplianceReceivingError as exc:
        db.session.rollback()
        flash(str(exc), "danger")

    return redirect(
        url_for(
            "appliance.receiving_detail",
            receiving_id=receiving_id,
        )
    )


# ============================================================
# Manager pricing after Posting
# ============================================================

@appliance_bp.post(
    "/receiving/lines/<int:line_id>/price"
)
@login_required
def receiving_price_line(line_id):
    line = db.session.get(
        ApplianceReceivingLine,
        line_id,
    )

    if line is None:
        flash(
            "Receiving line not found.",
            "danger",
        )
        return redirect(
            url_for(
                "appliance.receiving_missing_prices"
            )
        )

    receiving_id = line.receiving_id

    try:
        ApplianceReceivingService.update_posted_line_price(
            line_id=line_id,
            actor=current_user,
            unit_cost=request.form.get("unit_cost"),
            selling_price=request.form.get(
                "selling_price"
            ),
        )

        flash(
            f"Price saved for line #{line.line_no}.",
            "success",
        )

    except ApplianceReceivingError as exc:
        db.session.rollback()
        flash(str(exc), "danger")

    return redirect(
        url_for(
            "appliance.receiving_detail",
            receiving_id=receiving_id,
        )
    )


# ============================================================
# Appliance Inventory
# ============================================================

@appliance_bp.get("/inventory")
@login_required
def inventory_list():

    from collections import defaultdict

    from sqlalchemy import func, or_

    from models import (
        ApplianceCategory,
        ApplianceIssue,
        ApplianceIssueLine,
        ApplianceUnit,
        Warehouse,
    )

    allowed_warehouse_ids = _warehouse_ids_for(
        "appliance.view"
    )

    # --------------------------------------------------------
    # Filters
    # --------------------------------------------------------

    q = (
        request.args.get("q")
        or ""
    ).strip()

    # IMPORTANT:
    # Inventory is a complete physical registry.
    # Default is ALL, not AVAILABLE.
    status = (
        request.args.get("status")
        or "all"
    ).strip().lower()

    condition = (
        request.args.get("condition")
        or ""
    ).strip().lower()

    raw_warehouse_id = (
        request.args.get("warehouse_id")
        or ""
    ).strip()

    raw_category_id = (
        request.args.get("category_id")
        or ""
    ).strip()

    warehouse_id = None
    category_id = None

    try:
        if raw_warehouse_id:
            warehouse_id = int(
                raw_warehouse_id
            )
    except (TypeError, ValueError):
        warehouse_id = None

    try:
        if raw_category_id:
            category_id = int(
                raw_category_id
            )
    except (TypeError, ValueError):
        category_id = None

    if (
        warehouse_id is not None
        and warehouse_id not in allowed_warehouse_ids
    ):
        warehouse_id = None

    # --------------------------------------------------------
    # Empty access
    # --------------------------------------------------------

    if not allowed_warehouse_ids:

        return render_template(
            "appliance_inventory.html",
            units=[],
            grouped_units=[],
            warehouses=[],
            categories=_active_categories(),
            q=q,
            warehouse_id=None,
            category_id=None,
            status="all",
            condition=condition,
            can_pricing=False,
            total_count=0,
            available_count=0,
            issued_count=0,
            issue_info={},
        )

    # --------------------------------------------------------
    # Base registry query
    # --------------------------------------------------------

    query = (
        ApplianceUnit.query
        .filter(
            ApplianceUnit.warehouse_id.in_(
                allowed_warehouse_ids
            )
        )
    )

    if warehouse_id is not None:
        query = query.filter(
            ApplianceUnit.warehouse_id
            == warehouse_id
        )

    if category_id is not None:
        query = query.filter(
            ApplianceUnit.category_id
            == category_id
        )

    if condition:
        query = query.filter(
            ApplianceUnit.condition
            == condition
        )

    if (
        status
        and status != "all"
    ):
        query = query.filter(
            ApplianceUnit.status
            == status
        )

    # --------------------------------------------------------
    # Search
    #
    # Physical fields + warehouse Issue history.
    # --------------------------------------------------------

    if q:

        like = f"%{q}%"

        issue_unit_ids = (
            db.session.query(
                ApplianceIssueLine.appliance_unit_id
            )
            .join(
                ApplianceIssue,
                ApplianceIssue.id
                == ApplianceIssueLine.issue_id,
            )
            .filter(
                or_(
                    ApplianceIssue.issue_number.ilike(
                        like
                    ),
                    ApplianceIssue.work_order_number.ilike(
                        like
                    ),
                )
            )
        )

        # Technician username needs User.
        from models import User

        issue_unit_ids_by_tech = (
            db.session.query(
                ApplianceIssueLine.appliance_unit_id
            )
            .join(
                ApplianceIssue,
                ApplianceIssue.id
                == ApplianceIssueLine.issue_id,
            )
            .outerjoin(
                User,
                User.id
                == ApplianceIssue.technician_id,
            )
            .filter(
                User.username.ilike(
                    like
                )
            )
        )

        query = query.filter(
            or_(
                ApplianceUnit.inventory_number.ilike(
                    like
                ),
                ApplianceUnit.serial_number.ilike(
                    like
                ),
                ApplianceUnit.model_number.ilike(
                    like
                ),
                ApplianceUnit.brand.ilike(
                    like
                ),
                ApplianceUnit.description.ilike(
                    like
                ),
                ApplianceUnit.current_work_order_number.ilike(
                    like
                ),
                ApplianceUnit.id.in_(
                    issue_unit_ids
                ),
                ApplianceUnit.id.in_(
                    issue_unit_ids_by_tech
                ),
            )
        )

    # --------------------------------------------------------
    # Counts across complete accessible registry.
    #
    # Counts intentionally ignore Status filter so user always
    # knows total / available / issued inventory.
    # Warehouse/category/condition filters are respected.
    # --------------------------------------------------------

    count_query = (
        db.session.query(
            ApplianceUnit.status,
            func.count(
                ApplianceUnit.id
            ),
        )
        .filter(
            ApplianceUnit.warehouse_id.in_(
                allowed_warehouse_ids
            )
        )
    )

    if warehouse_id is not None:
        count_query = count_query.filter(
            ApplianceUnit.warehouse_id
            == warehouse_id
        )

    if category_id is not None:
        count_query = count_query.filter(
            ApplianceUnit.category_id
            == category_id
        )

    if condition:
        count_query = count_query.filter(
            ApplianceUnit.condition
            == condition
        )

    count_rows = (
        count_query
        .group_by(
            ApplianceUnit.status
        )
        .all()
    )

    status_counts = {
        (row[0] or "").lower():
            int(row[1] or 0)
        for row in count_rows
    }

    registry_total_count = sum(
        status_counts.values()
    )

    available_count = status_counts.get(
        "available",
        0,
    )

    issued_count = status_counts.get(
        "issued",
        0,
    )

    # --------------------------------------------------------
    # Load visible rows
    # --------------------------------------------------------

    units = (
        query
        .order_by(
            ApplianceUnit.category_id.asc(),
            ApplianceUnit.brand.asc(),
            ApplianceUnit.model_number.asc(),
            ApplianceUnit.inventory_number.asc(),
        )
        .limit(500)
        .all()
    )

    visible_unit_ids = [
        unit.id
        for unit in units
    ]

    # --------------------------------------------------------
    # Latest Issue info for each physical unit
    #
    # We do NOT store this directly on ApplianceUnit because
    # Issue/Movement history is authoritative.
    # --------------------------------------------------------

    issue_info = {}

    if visible_unit_ids:

        issue_rows = (
            db.session.query(
                ApplianceIssueLine,
                ApplianceIssue,
            )
            .join(
                ApplianceIssue,
                ApplianceIssue.id
                == ApplianceIssueLine.issue_id,
            )
            .filter(
                ApplianceIssueLine.appliance_unit_id.in_(
                    visible_unit_ids
                )
            )
            .order_by(
                ApplianceIssueLine.appliance_unit_id.asc(),
                ApplianceIssue.issued_at.desc(),
                ApplianceIssue.id.desc(),
            )
            .all()
        )

        for line, issue in issue_rows:

            unit_id = int(
                line.appliance_unit_id
            )

            # First row is newest because of ORDER BY.
            if unit_id in issue_info:
                continue

            issue_info[unit_id] = {
                "issue_id":
                    issue.id,

                "issue_number":
                    issue.issue_number,

                "technician":
                    (
                        issue.technician.username
                        if issue.technician
                        else ""
                    ),

                "work_order_number":
                    (
                        line.current_work_order_number
                        or issue.work_order_number
                        or ""
                    ),

                "issued_at":
                    issue.issued_at_local,
            }

    # --------------------------------------------------------
    # Group visible units by category.
    # Keep existing UI grouping.
    # --------------------------------------------------------

    grouped_map = defaultdict(list)

    for unit in units:

        grouped_map[
            unit.category_id
        ].append(
            unit
        )

    grouped_units = []

    categories_by_id = {
        category.id: category
        for category in _active_categories()
    }

    for category_id_key, rows in grouped_map.items():

        category = categories_by_id.get(
            category_id_key
        )

        if category is None and rows:
            category = rows[0].category

        grouped_units.append(
            {
                "category": category,
                "units": rows,
            }
        )

    grouped_units.sort(
        key=lambda group: (
            (
                group["category"].sort_order
                if group["category"] is not None
                else 999999
            ),
            (
                group["category"].name
                if group["category"] is not None
                else ""
            ),
        )
    )

    # --------------------------------------------------------
    # Visible warehouses
    # --------------------------------------------------------

    warehouses = (
        Warehouse.query
        .filter(
            Warehouse.id.in_(
                allowed_warehouse_ids
            ),
            Warehouse.is_active.is_(True),
        )
        .order_by(
            Warehouse.code.asc()
        )
        .all()
    )

    # --------------------------------------------------------
    # Pricing permission
    # --------------------------------------------------------

    can_pricing = any(
        AccessControlService.can(
            current_user,
            "appliance.pricing",
            warehouse_id=warehouse_id_value,
        )
        for warehouse_id_value
        in allowed_warehouse_ids
    )

    return render_template(
        "appliance_inventory.html",

        units=units,
        grouped_units=grouped_units,

        warehouses=warehouses,
        categories=_active_categories(),

        q=q,
        warehouse_id=warehouse_id,
        category_id=category_id,
        status=status,
        condition=condition,

        can_pricing=can_pricing,

        # total_count = complete filtered registry,
        # NOT only currently visible status.
        total_count=registry_total_count,
        available_count=available_count,
        issued_count=issued_count,

        issue_info=issue_info,
    )



@appliance_bp.get(
    "/inventory/<int:unit_id>"
)
@login_required
def inventory_detail(unit_id):
    unit = db.session.get(
        ApplianceUnit,
        unit_id,
    )

    if unit is None:
        flash(
            "Appliance unit not found.",
            "danger",
        )
        return redirect(
            url_for("appliance.inventory_list")
        )

    if not AccessControlService.can(
        current_user,
        "appliance.view",
        warehouse_id=unit.warehouse_id,
    ):
        flash(
            "Access denied.",
            "danger",
        )
        return redirect(
            url_for("appliance.inventory_list")
        )

    can_pricing = AccessControlService.can(
        current_user,
        "appliance.pricing",
        warehouse_id=unit.warehouse_id,
    )

    return render_template(
        "appliance_inventory_detail.html",
        unit=unit,
        can_pricing=can_pricing,
    )


# ============================================================
# Appliance Issue UI
# Warehouse Issue Slip - NO PRICES
# ============================================================

from services.appliance_issue_service import (
    ApplianceIssueAccessDenied,
    ApplianceIssueError,
    ApplianceIssueService,
)


def _issue_allowed_warehouses():
    """
    Warehouses where current user may perform
    Appliance warehouse operations.
    """
    return [
        warehouse
        for warehouse in AccessControlService.accessible_warehouses(
            current_user
        )
        if AccessControlService.can(
            current_user,
            "appliance.receive",
            warehouse_id=warehouse.id,
        )
    ]


# ============================================================
# NEW ISSUE
# ============================================================

@appliance_bp.route(
    "/issues/new",
    methods=["GET", "POST"],
)
@login_required
def issue_new():

    from models import User

    warehouses = _issue_allowed_warehouses()

    if not warehouses:
        flash(
            "You do not have access to issue appliances.",
            "danger",
        )
        return redirect(
            url_for("appliance.inventory_list")
        )

    technicians = (
        User.query
        .filter(
            User.role == "technician"
        )
        .order_by(
            User.username.asc()
        )
        .all()
    )

    if request.method == "POST":

        try:
            warehouse_id = int(
                request.form.get("warehouse_id")
                or 0
            )

            technician_id = int(
                request.form.get("technician_id")
                or 0
            )

            work_order_number = (
                request.form.get("work_order_number")
                or ""
            ).strip().upper()

            unit_ids = []

            for raw in request.form.getlist(
                "appliance_unit_ids"
            ):
                try:
                    value = int(raw)
                except (TypeError, ValueError):
                    continue

                if value > 0:
                    unit_ids.append(value)

            issue = ApplianceIssueService.create_issue(
                actor=current_user,
                warehouse_id=warehouse_id,
                technician_id=technician_id,
                work_order_number=work_order_number,
                appliance_unit_ids=unit_ids,
                notes=request.form.get("notes"),
            )

            flash(
                f"{issue.issue_number} created successfully.",
                "success",
            )

            return redirect(
                url_for(
                    "appliance.issue_detail",
                    issue_id=issue.id,
                    print=1,
                )
            )

        except (
            ApplianceIssueAccessDenied,
            ApplianceIssueError,
            ValueError,
            TypeError,
        ) as exc:
            db.session.rollback()

            flash(
                str(exc),
                "danger",
            )

    default_warehouse = None

    try:
        candidate = (
            AccessControlService.default_warehouse(
                current_user
            )
        )

        if (
            candidate is not None
            and any(
                int(row.id) == int(candidate.id)
                for row in warehouses
            )
        ):
            default_warehouse = candidate

    except Exception:
        default_warehouse = None

    if default_warehouse is None:
        default_warehouse = warehouses[0]

    return render_template(
        "appliance_issue_new.html",
        warehouses=warehouses,
        technicians=technicians,
        default_warehouse_id=(
            default_warehouse.id
            if default_warehouse
            else None
        ),
    )


# ============================================================
# Search AVAILABLE appliances for Issue
# ============================================================

@appliance_bp.get(
    "/issues/search-appliances"
)
@login_required
def issue_search_appliances():
    """
    AVAILABLE appliance catalog for warehouse Issue UI.

    Supports:
        warehouse_id
        category_id
        brand
        q

    Response includes:
        category counts
        brands
        filtered physical appliance units

    NO pricing is returned.
    """

    from sqlalchemy import func, or_

    from models import (
        ApplianceCategory,
        ApplianceUnit,
    )

    # --------------------------------------------------------
    # Warehouse
    # --------------------------------------------------------

    try:
        warehouse_id = int(
            request.args.get("warehouse_id")
            or 0
        )
    except (TypeError, ValueError):
        warehouse_id = 0

    if warehouse_id <= 0:
        return jsonify(
            {
                "ok": True,
                "categories": [],
                "brands": [],
                "items": [],
                "total_available": 0,
            }
        )

    if not AccessControlService.can(
        current_user,
        "appliance.receive",
        warehouse_id=warehouse_id,
    ):
        return jsonify(
            {
                "ok": False,
                "error": "Access denied.",
            }
        ), 403

    # --------------------------------------------------------
    # Filters
    # --------------------------------------------------------

    try:
        category_id = int(
            request.args.get("category_id")
            or 0
        )
    except (TypeError, ValueError):
        category_id = 0

    brand = (
        request.args.get("brand")
        or ""
    ).strip().upper()

    q = (
        request.args.get("q")
        or ""
    ).strip()

    # --------------------------------------------------------
    # Category counters
    #
    # Always based on full AVAILABLE inventory in warehouse,
    # independent of currently selected category/brand/search.
    # --------------------------------------------------------

    category_rows = (
        db.session.query(
            ApplianceCategory.id,
            ApplianceCategory.name,
            func.count(
                ApplianceUnit.id
            ).label("unit_count"),
        )
        .join(
            ApplianceUnit,
            ApplianceUnit.category_id
            == ApplianceCategory.id,
        )
        .filter(
            ApplianceUnit.warehouse_id
            == warehouse_id,
            ApplianceUnit.status
            == "available",
        )
        .group_by(
            ApplianceCategory.id,
            ApplianceCategory.name,
        )
        .order_by(
            ApplianceCategory.name.asc()
        )
        .all()
    )

    categories = [
        {
            "id": row.id,
            "name": (
                row.name
                or ""
            ).upper(),
            "count": int(
                row.unit_count
                or 0
            ),
        }
        for row in category_rows
    ]

    total_available = sum(
        row["count"]
        for row in categories
    )

    # --------------------------------------------------------
    # Brands
    #
    # Brand list follows selected category, so after clicking
    # REFRIGERATOR we only show refrigerator brands.
    # --------------------------------------------------------

    brand_query = (
        db.session.query(
            ApplianceUnit.brand
        )
        .filter(
            ApplianceUnit.warehouse_id
            == warehouse_id,
            ApplianceUnit.status
            == "available",
            ApplianceUnit.brand.isnot(None),
            func.trim(
                ApplianceUnit.brand
            ) != "",
        )
    )

    if category_id > 0:
        brand_query = brand_query.filter(
            ApplianceUnit.category_id
            == category_id
        )

    brand_rows = (
        brand_query
        .distinct()
        .order_by(
            ApplianceUnit.brand.asc()
        )
        .all()
    )

    brands = sorted(
        {
            (
                row[0]
                or ""
            ).strip().upper()
            for row in brand_rows
            if (
                row[0]
                or ""
            ).strip()
        }
    )

    # --------------------------------------------------------
    # Physical AVAILABLE units
    # --------------------------------------------------------

    query = (
        ApplianceUnit.query
        .join(
            ApplianceCategory,
            ApplianceUnit.category_id
            == ApplianceCategory.id,
        )
        .filter(
            ApplianceUnit.warehouse_id
            == warehouse_id,
            ApplianceUnit.status
            == "available",
        )
    )

    if category_id > 0:
        query = query.filter(
            ApplianceUnit.category_id
            == category_id
        )

    if brand:
        query = query.filter(
            func.upper(
                func.trim(
                    ApplianceUnit.brand
                )
            )
            == brand
        )

    if q:
        like = f"%{q}%"

        query = query.filter(
            or_(
                ApplianceUnit.inventory_number.ilike(
                    like
                ),
                ApplianceUnit.serial_number.ilike(
                    like
                ),
                ApplianceUnit.model_number.ilike(
                    like
                ),
                ApplianceUnit.brand.ilike(
                    like
                ),
                ApplianceCategory.name.ilike(
                    like
                ),
            )
        )

    units = (
        query
        .order_by(
            ApplianceCategory.name.asc(),
            ApplianceUnit.brand.asc(),
            ApplianceUnit.model_number.asc(),
            ApplianceUnit.serial_number.asc(),
            ApplianceUnit.inventory_number.asc(),
        )
        .limit(100)
        .all()
    )

    items = []

    for unit in units:

        category_name = (
            unit.category.name
            if unit.category
            else ""
        )

        size_text = ""

        if unit.size_value is not None:
            size_number = f"{unit.size_value:g}"

            size_text = (
                f"{size_number} "
                f"{unit.size_unit or ''}"
            ).strip()

        items.append(
            {
                "id": unit.id,

                "inventory_number": (
                    unit.inventory_number
                    or ""
                ).upper(),

                "appliance_type": (
                    category_name
                    or ""
                ).upper(),

                "category_id":
                    unit.category_id,

                "brand": (
                    unit.brand
                    or ""
                ).upper(),

                "model": (
                    unit.model_number
                    or ""
                ).upper(),

                "serial": (
                    unit.serial_number
                    or ""
                ).upper(),

                "size":
                    size_text.upper(),

                "condition": (
                    unit.condition
                    or ""
                ).replace(
                    "_",
                    " ",
                ).upper(),
            }
        )

    return jsonify(
        {
            "ok": True,
            "categories": categories,
            "brands": brands,
            "items": items,
            "total_available":
                total_available,
        }
    )


# ============================================================
# Search Work Orders
# ============================================================

@appliance_bp.get(
    "/issues/search-work-orders"
)
@login_required
def issue_search_work_orders():

    from sqlalchemy import or_
    from models import WorkOrder

    q = (
        request.args.get("q")
        or ""
    ).strip()

    if not q:
        return jsonify(
            {
                "ok": True,
                "items": [],
            }
        )

    like = f"%{q}%"

    work_orders = (
        WorkOrder.query
        .filter(
            or_(
                WorkOrder.job_numbers.ilike(
                    like
                ),
                WorkOrder.technician_name.ilike(
                    like
                ),
                WorkOrder.customer_po.ilike(
                    like
                ),
            )
        )
        .order_by(
            WorkOrder.id.desc()
        )
        .limit(25)
        .all()
    )

    return jsonify(
        {
            "ok": True,
            "items": [
                {
                    "id": wo.id,

                    "job_numbers":
                        (
                            wo.job_numbers
                            or ""
                        ).upper(),

                    "canonical_job":
                        (
                            wo.canonical_job
                            or ""
                        ).upper(),

                    "technician":
                        (
                            wo.technician_username
                            or ""
                        ).upper(),

                    "customer_po":
                        (
                            wo.customer_po
                            or ""
                        ).upper(),
                }
                for wo in work_orders
            ],
        }
    )


# ============================================================
# ISSUE DETAIL / PRINT SLIP
# ============================================================

@appliance_bp.get(
    "/issues/<int:issue_id>"
)
@login_required
def issue_detail(issue_id):

    try:
        issue = ApplianceIssueService.get_issue(
            issue_id=issue_id,
            actor=current_user,
        )

    except (
        ApplianceIssueAccessDenied,
        ApplianceIssueError,
    ) as exc:

        flash(
            str(exc),
            "danger",
        )

        return redirect(
            url_for("appliance.inventory_list")
        )

    auto_print = (
        request.args.get("print")
        == "1"
    )

    return render_template(
        "appliance_issue_detail.html",
        issue=issue,
        auto_print=auto_print,
    )


# ============================================================
# DELETE APPLIANCE ISSUE
#
# Superadmin-only correction tool.
#
# This is NOT a normal warehouse return.
# It removes an incorrectly created Issue only while the
# appliances are still untouched after the original ISSUE.
# ============================================================

@appliance_bp.post(
    "/issues/<int:issue_id>/delete"
)
@login_required
def issue_delete(issue_id):

    from models import (
        ApplianceIssue,
        ApplianceIssueLine,
        ApplianceMovement,
        ApplianceUnit,
    )

    # --------------------------------------------------------
    # Role protection
    # --------------------------------------------------------

    if (
        getattr(current_user, "role", "")
        or ""
    ).strip().lower() != "superadmin":

        flash(
            "Only SUPERADMIN can delete an Appliance Issue.",
            "danger",
        )

        return redirect(
            url_for(
                "appliance.issue_detail",
                issue_id=issue_id,
            )
        )

    issue = db.session.get(
        ApplianceIssue,
        issue_id,
    )

    if issue is None:

        flash(
            "Appliance Issue not found.",
            "danger",
        )

        return redirect(
            url_for(
                "appliance.inventory_list"
            )
        )

    # User must also have warehouse access.
    if not AccessControlService.can(
        current_user,
        "appliance.receive",
        warehouse_id=issue.warehouse_id,
    ):

        flash(
            "You do not have access to this warehouse.",
            "danger",
        )

        return redirect(
            url_for(
                "appliance.inventory_list"
            )
        )

    try:

        # ----------------------------------------------------
        # Lines / physical units
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Find original ISSUE movements belonging to this AIS.
        # ----------------------------------------------------

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

        original_movement_ids = [
            movement.id
            for movement
            in original_issue_movements
        ]

        # ----------------------------------------------------
        # SAFETY CHECK:
        #
        # If ANY other movement already exists for one of these
        # physical appliances, normal DELETE is no longer valid.
        #
        # Future examples:
        # RETURN
        # REPLACE
        # CHANGE_WO
        # INSTALLED
        # EXCHANGE
        # ----------------------------------------------------

        later_query = (
            ApplianceMovement.query
            .filter(
                ApplianceMovement.appliance_unit_id.in_(
                    unit_ids
                )
            )
        )

        if original_movement_ids:

            later_query = later_query.filter(
                ~ApplianceMovement.id.in_(
                    original_movement_ids
                )
            )

        other_movement = (
            later_query
            .order_by(
                ApplianceMovement.created_at.desc()
            )
            .first()
        )

        if other_movement is not None:

            raise ApplianceIssueError(
                "This Issue cannot be deleted because "
                "one or more appliances already have "
                "additional movement history. "
                "Use RETURN / REPLACE / CHANGE W/O instead."
            )

        # ----------------------------------------------------
        # Current-state validation
        # ----------------------------------------------------

        for line in lines:

            unit = units_by_id[
                int(line.appliance_unit_id)
            ]

            status = (
                unit.status
                or ""
            ).strip().lower()

            if status != "issued":

                raise ApplianceIssueError(
                    f"{unit.inventory_number} is currently "
                    f"{status.upper() or 'UNKNOWN'}. "
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

        # ----------------------------------------------------
        # Restore inventory
        # ----------------------------------------------------

        now = datetime.utcnow()

        for unit in units:

            unit.status = "available"

            unit.current_work_order_id = None
            unit.current_work_order_number = None

            unit.updated_at = now
            unit.updated_by_id = current_user.id

        db.session.flush()

        # ----------------------------------------------------
        # Remove original ISSUE movements
        # ----------------------------------------------------

        (
            ApplianceMovement.query
            .filter(
                ApplianceMovement.issue_id
                == issue.id
            )
            .delete(
                synchronize_session=False
            )
        )

        # ----------------------------------------------------
        # Remove Issue Lines
        # ----------------------------------------------------

        (
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

        # ----------------------------------------------------
        # Remove Issue header
        #
        # Bulk delete avoids SQLAlchemy trying to cascade-delete
        # already deleted lines a second time.
        # ----------------------------------------------------

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

        db.session.commit()

        flash(
            f"{issue_number} deleted. "
            f"{len(units)} appliance(s) returned to AVAILABLE.",
            "success",
        )

        return redirect(
            url_for(
                "appliance.inventory_list"
            )
        )

    except (
        ApplianceIssueError,
        ValueError,
    ) as exc:

        db.session.rollback()

        flash(
            str(exc),
            "danger",
        )

    except Exception:

        db.session.rollback()
        raise

    return redirect(
        url_for(
            "appliance.issue_detail",
            issue_id=issue_id,
        )
    )


