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
        "color": request.form.get("color"),
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

    delete_draft_warehouse_ids = set(
        _warehouse_ids_for(
            "appliance.receiving.delete_draft"
        )
    )

    can_delete_draft_ids = {
        receiving.id
        for receiving in receivings
        if (
            receiving.status == "draft"
            and receiving.warehouse_id
            in delete_draft_warehouse_ids
        )
    }

    return render_template(
        "appliance_receiving_list.html",
        receivings=receivings,
        missing_ids=missing_ids,
        can_delete_draft_ids=can_delete_draft_ids,
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

    can_void = AccessControlService.can(
        current_user,
        "appliance.receiving.void",
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
        can_void=can_void,
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
# Receiving - Add Appliance Type
#
# Operational master-data action.
#
# Any warehouse user who has appliance.receive may create
# a new ApplianceCategory directly during Receiving.
#
# No schema change is required.
# ============================================================

@appliance_bp.post(
    "/receiving/categories/add"
)
@login_required
def receiving_add_category():

    import re
    from datetime import datetime

    # --------------------------------------------------------
    # Permission
    #
    # Category is global master data, but creation is an
    # operational Receiving action. User must have
    # appliance.receive in at least one accessible warehouse.
    # --------------------------------------------------------

    allowed_warehouse_ids = _warehouse_ids_for(
        "appliance.receive"
    )

    if not allowed_warehouse_ids:
        flash(
            "Access denied.",
            "danger",
        )

        return redirect(
            url_for(
                "appliance.receiving_list"
            )
        )

    # --------------------------------------------------------
    # Normalize NAME
    # --------------------------------------------------------

    raw_name = (
        request.form.get("name")
        or ""
    )

    name = " ".join(
        raw_name.strip().upper().split()
    )

    if not name:
        flash(
            "Appliance Type name is required.",
            "danger",
        )

        return redirect(
            request.referrer
            or url_for(
                "appliance.receiving_list"
            )
        )

    if len(name) > 120:
        flash(
            "Appliance Type name cannot exceed "
            "120 characters.",
            "danger",
        )

        return redirect(
            request.referrer
            or url_for(
                "appliance.receiving_list"
            )
        )

    # --------------------------------------------------------
    # Build stable CODE automatically
    #
    # Example:
    #   Ice Maker      -> ICE_MAKER
    #   Mini Split A/C -> MINI_SPLIT_A_C
    #
    # code max length = 40
    # --------------------------------------------------------

    code = re.sub(
        r"[^A-Z0-9]+",
        "_",
        name,
    ).strip("_")

    if not code:
        flash(
            "Unable to generate Appliance Type code.",
            "danger",
        )

        return redirect(
            request.referrer
            or url_for(
                "appliance.receiving_list"
            )
        )

    code = code[:40].rstrip("_")

    # --------------------------------------------------------
    # Duplicate NAME check - case insensitive
    # --------------------------------------------------------

    from sqlalchemy import func

    existing_by_name = (
        ApplianceCategory.query
        .filter(
            func.upper(
                func.trim(
                    ApplianceCategory.name
                )
            ) == name
        )
        .first()
    )

    now = datetime.utcnow()

    if existing_by_name is not None:

        # Existing active category -> simply tell the user.
        if existing_by_name.is_active:

            flash(
                f"Appliance Type "
                f"'{existing_by_name.name}' "
                f"already exists.",
                "info",
            )

            return redirect(
                request.referrer
                or url_for(
                    "appliance.receiving_list"
                )
            )

        # Existing but inactive category:
        # reactivate instead of creating duplicate master data.
        try:
            existing_by_name.is_active = True
            existing_by_name.updated_at = now
            existing_by_name.updated_by_id = (
                current_user.id
            )

            db.session.commit()

            flash(
                f"Appliance Type "
                f"'{existing_by_name.name}' "
                f"was reactivated.",
                "success",
            )

        except Exception:
            db.session.rollback()
            raise

        return redirect(
            request.referrer
            or url_for(
                "appliance.receiving_list"
            )
        )

    # --------------------------------------------------------
    # CODE collision
    #
    # Different names can normalize to same first 40 chars.
    # Generate suffix _2, _3, ...
    # --------------------------------------------------------

    base_code = code
    suffix = 2

    while (
        ApplianceCategory.query
        .filter(
            func.upper(
                ApplianceCategory.code
            ) == code
        )
        .first()
        is not None
    ):

        suffix_text = f"_{suffix}"

        max_base_length = (
            40 - len(suffix_text)
        )

        code = (
            base_code[:max_base_length]
            .rstrip("_")
            + suffix_text
        )

        suffix += 1

        if suffix > 9999:
            flash(
                "Unable to generate unique "
                "Appliance Type code.",
                "danger",
            )

            return redirect(
                request.referrer
                or url_for(
                    "appliance.receiving_list"
                )
            )

    # --------------------------------------------------------
    # Create active category
    # --------------------------------------------------------

    category = ApplianceCategory(
        code=code,
        name=name,
        description=None,
        sort_order=100,
        is_active=True,
        created_at=now,
        created_by_id=current_user.id,
        updated_at=now,
        updated_by_id=current_user.id,
    )

    try:
        db.session.add(
            category
        )

        db.session.commit()

        flash(
            f"Appliance Type '{name}' added.",
            "success",
        )

    except Exception:
        db.session.rollback()
        raise

    # Reload the exact Receiving page the warehouse worker
    # came from. _active_categories() will now include the
    # newly created category in every dropdown.
    return redirect(
        request.referrer
        or url_for(
            "appliance.receiving_list"
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
# Receiving Model Specs Cache
#
# Exact Brand + Model lookup in the persistent local
# ApplianceModelSpec catalog.
#
# IMPORTANT:
# - LOCAL DB ONLY;
# - no OpenAI;
# - no Web Search;
# - read-only;
# - only confirmed model specs are returned.
# ============================================================

@appliance_bp.get(
    "/receiving/model-specs-cache"
)
@login_required
def receiving_model_specs_cache():

    from flask import current_app
    from models import ApplianceModelSpec

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
    # Exact normalized Brand + Model
    #
    # Must use exactly the same normalization rule as the
    # persistent model catalog.
    # --------------------------------------------------------

    brand = (
        request.args.get("brand")
        or ""
    )

    model_number = (
        request.args.get("model_number")
        or ""
    )

    normalized_brand = " ".join(
        brand.strip().upper().split()
    )

    normalized_model = " ".join(
        model_number.strip().upper().split()
    )

    if (
        not normalized_brand
        or not normalized_model
    ):
        return jsonify(
            {
                "ok": True,
                "hit": False,
            }
        )

    # --------------------------------------------------------
    # Confirmed local catalog only
    # --------------------------------------------------------

    spec = (
        ApplianceModelSpec.query
        .filter(
            ApplianceModelSpec.normalized_brand
            == normalized_brand,

            ApplianceModelSpec.normalized_model
            == normalized_model,

            ApplianceModelSpec.exact_model_confirmed
            .is_(True),
        )
        .first()
    )

    if spec is None:
        return jsonify(
            {
                "ok": True,
                "hit": False,
            }
        )

    current_app.logger.info(
        "RECEIVING_MODEL_SPECS_CACHE_HIT "
        "brand=%s model=%s cache_id=%s",
        normalized_brand,
        normalized_model,
        spec.id,
    )

    return jsonify(
        {
            "ok": True,
            "hit": True,
            "result": {
                "id": spec.id,
                "brand": spec.brand,
                "model_number":
                    spec.model_number,
                "size_value":
                    spec.size_value,
                "size_unit":
                    spec.size_unit,
                "color":
                    spec.color,
                "notes_block":
                    spec.notes_block,
            },
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
# Delete Draft Receiving
# ============================================================

@appliance_bp.post(
    "/receiving/<int:receiving_id>/delete"
)
@login_required
def receiving_delete_draft(receiving_id):
    try:
        ApplianceReceivingService.delete_draft(
            receiving_id=receiving_id,
            actor=current_user,
        )

        flash(
            "Draft Appliance Receiving deleted.",
            "success",
        )

        return redirect(
            url_for("appliance.receiving_list")
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
# Emergency Purge Receiving
#
# BREAK-GLASS operation.
# SUPERADMIN only.
# Not controlled by assignable Permission catalog.
# ============================================================

@appliance_bp.post(
    "/receiving/<int:receiving_id>/emergency-purge"
)
@login_required
def receiving_emergency_purge(receiving_id):
    # --------------------------------------------------------
    # Defense in depth:
    # route itself is SUPERADMIN-only.
    #
    # The service repeats this check and performs the full
    # dependency / state / confirmation validation.
    # --------------------------------------------------------

    role = (
        getattr(current_user, "role", "")
        or ""
    ).strip().lower()

    if role != "superadmin":
        flash(
            "Emergency Purge is restricted to SUPERADMIN.",
            "danger",
        )

        return redirect(
            url_for(
                "appliance.receiving_detail",
                receiving_id=receiving_id,
            )
        )

    try:
        result = (
            ApplianceReceivingService.emergency_purge(
                receiving_id=receiving_id,
                actor=current_user,
                confirmation=request.form.get(
                    "confirmation"
                ),
            )
        )

        flash(
            (
                f'{result["receiving_number"]} permanently '
                f'purged. '
                f'{result["deleted_units"]} appliance unit(s) '
                f'and {result["deleted_movements"]} technical '
                f'movement(s) removed.'
            ),
            "warning",
        )

        # Receiving no longer exists after successful purge.
        return redirect(
            url_for("appliance.receiving_list")
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
# Void Posted Receiving
# ============================================================

@appliance_bp.post(
    "/receiving/<int:receiving_id>/void"
)
@login_required
def receiving_void(receiving_id):
    try:
        receiving = (
            ApplianceReceivingService.void_receiving(
                receiving_id=receiving_id,
                actor=current_user,
                reason=request.form.get("reason"),
            )
        )

        flash(
            f"{receiving.receiving_number} voided.",
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
# Posted Receiving line editing
#
# appliance.receive  -> Size / Unit / Color / Notes
# appliance.pricing  -> Cost / Sell
# ============================================================

@appliance_bp.post(
    "/receiving/lines/<int:line_id>/posted-update"
)
@login_required
def receiving_posted_line_update(line_id):

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
                "appliance.receiving_list"
            )
        )

    receiving_id = line.receiving_id

    # Presence of these fields tells the service which
    # permission must be enforced.
    detail_fields = {
        "size_value",
        "size_unit",
        "color",
        "notes",
    }

    pricing_fields = {
        "unit_cost",
        "selling_price",
    }

    update_details = any(
        field in request.form
        for field in detail_fields
    )

    update_pricing = any(
        field in request.form
        for field in pricing_fields
    )

    try:

        ApplianceReceivingService.update_posted_line_fields(
            line_id=line_id,
            actor=current_user,

            update_details=update_details,
            update_pricing=update_pricing,

            size_value=request.form.get(
                "size_value"
            ),
            size_unit=request.form.get(
                "size_unit"
            ),
            color=request.form.get(
                "color"
            ),
            notes=request.form.get(
                "notes"
            ),

            unit_cost=request.form.get(
                "unit_cost"
            ),
            selling_price=request.form.get(
                "selling_price"
            ),
        )

        if update_details and update_pricing:
            message = (
                f"Details and price saved "
                f"for line #{line.line_no}."
            )

        elif update_details:
            message = (
                f"Details saved "
                f"for line #{line.line_no}."
            )

        else:
            message = (
                f"Price saved "
                f"for line #{line.line_no}."
            )

        flash(
            message,
            "success",
        )

    except ApplianceReceivingError as exc:
        db.session.rollback()
        flash(
            str(exc),
            "danger",
        )

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
# Appliance Categories Management
# ============================================================

def _can_manage_appliance_categories():
    """
    Global ApplianceCategory master-data management.

    Warehouse Receiving staff may CREATE missing categories
    from Receiving through appliance.receive.

    Activation / deactivation is intentionally restricted to
    management roles because ApplianceCategory is global data.
    """
    return (
        current_user.is_authenticated
        and current_user.role in (
            "superadmin",
            "admin",
            "manager",
        )
    )


@appliance_bp.get("/categories")
@login_required
def categories_list():

    if not _can_manage_appliance_categories():
        flash(
            "Access denied.",
            "danger",
        )
        return redirect(
            url_for(
                "appliance.inventory_list"
            )
        )

    categories = (
        ApplianceCategory.query
        .order_by(
            ApplianceCategory.is_active.desc(),
            ApplianceCategory.sort_order.asc(),
            ApplianceCategory.name.asc(),
        )
        .all()
    )

    return render_template(
        "appliance_categories.html",
        categories=categories,
    )


@appliance_bp.post(
    "/categories/<int:category_id>/toggle"
)
@login_required
def categories_toggle(category_id):

    if not _can_manage_appliance_categories():
        flash(
            "Access denied.",
            "danger",
        )
        return redirect(
            url_for(
                "appliance.inventory_list"
            )
        )

    category = (
        ApplianceCategory.query
        .filter(
            ApplianceCategory.id == category_id
        )
        .first()
    )

    if category is None:
        flash(
            "Appliance type not found.",
            "danger",
        )
        return redirect(
            url_for(
                "appliance.categories_list"
            )
        )

    old_status = bool(
        category.is_active
    )

    category.is_active = not old_status

    now = datetime.utcnow()

    if hasattr(
        category,
        "updated_at",
    ):
        category.updated_at = now

    if hasattr(
        category,
        "updated_by_id",
    ):
        category.updated_by_id = (
            current_user.id
        )

    try:
        db.session.commit()

    except Exception:
        db.session.rollback()
        raise

    if category.is_active:
        flash(
            f"{category.name} activated.",
            "success",
        )
    else:
        flash(
            f"{category.name} deactivated. "
            "Historical Receiving and Inventory records were not changed.",
            "warning",
        )

    return redirect(
        url_for(
            "appliance.categories_list"
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

    # --------------------------------------------------------
    # Inventory purpose/classification.
    #
    # This is independent from STATUS and CONDITION.
    # --------------------------------------------------------
    stock_class = (
        request.args.get("stock_class")
        or ""
    ).strip().lower()

    allowed_stock_classes = {
        "new",
        "loaner",
        "retail",
    }

    if stock_class not in allowed_stock_classes:
        stock_class = ""

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
            stock_class=stock_class,
            can_pricing=False,
            total_count=0,
            available_count=0,
            issued_count=0,
            vendor_return_pending_count=0,
            issue_info={},
            issue_history={},
            repair_history={},
        )

    # --------------------------------------------------------
    # Manager control:
    # appliances waiting for physical vendor return.
    #
    # Count is warehouse-access aware and independent from
    # Inventory screen filters.
    # --------------------------------------------------------

    vendor_return_pending_count = (
        db.session.query(
            func.count(
                ApplianceUnit.id
            )
        )
        .filter(
            ApplianceUnit.warehouse_id.in_(
                allowed_warehouse_ids
            ),
            ApplianceUnit.status
            == "vendor_return_pending",
        )
        .scalar()
        or 0
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

    if stock_class:
        query = query.filter(
            ApplianceUnit.stock_class
            == stock_class
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
    issue_history = {}

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

            history_row = {
                "issue_id":
                    issue.id,

                "issue_number":
                    issue.issue_number,

                "line_status":
                    (
                        line.status
                        or ""
                    ).strip().lower(),

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

                "resolved_at":
                    (
                        line.resolved_at
                        if line.resolved_at
                        else None
                    ),
            }

            issue_history.setdefault(
                unit_id,
                [],
            ).append(
                history_row
            )

            # First row is newest because of ORDER BY.
            if unit_id not in issue_info:

                issue_info[unit_id] = {
                    **history_row,
                }

    # --------------------------------------------------------
    # Repair Order history for each visible physical appliance.
    #
    # RepairOrder is historical domain data and must remain
    # visible even after the appliance returns to AVAILABLE.
    #
    # One bulk query for all visible units; no N+1 queries.
    # --------------------------------------------------------

    repair_history = {}

    if visible_unit_ids:

        from models import ApplianceRepairOrder

        repair_rows = (
            ApplianceRepairOrder.query
            .filter(
                ApplianceRepairOrder.appliance_unit_id.in_(
                    visible_unit_ids
                )
            )
            .order_by(
                ApplianceRepairOrder.appliance_unit_id.asc(),
                ApplianceRepairOrder.sent_at.desc(),
                ApplianceRepairOrder.id.desc(),
            )
            .all()
        )

        for repair in repair_rows:

            unit_id = int(
                repair.appliance_unit_id
            )

            repair_history.setdefault(
                unit_id,
                [],
            ).append(
                {
                    "repair_id":
                        repair.id,

                    "repair_number":
                        repair.repair_number,

                    "status":
                        (
                            repair.status
                            or ""
                        ).strip().lower(),
                }
            )

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
        vendor_return_pending_count=vendor_return_pending_count,

        issue_info=issue_info,
        issue_history=issue_history,
        repair_history=repair_history,
    )



@appliance_bp.get(
    "/inventory/<int:unit_id>"
)
@login_required
def inventory_detail(unit_id):

    import json

    from models import ApplianceMovement, User

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

    # Warehouse descriptive data may be corrected from the
    # physical Appliance Inventory record.
    can_receive = AccessControlService.can(
        current_user,
        "appliance.receive",
        warehouse_id=unit.warehouse_id,
    )

    # --------------------------------------------------------
    # Complete movement history for this physical appliance.
    # Newest first for operational review.
    # --------------------------------------------------------

    movements = (
        ApplianceMovement.query
        .filter(
            ApplianceMovement.appliance_unit_id
            == unit.id
        )
        .order_by(
            ApplianceMovement.created_at.desc(),
            ApplianceMovement.id.desc(),
        )
        .all()
    )

    # --------------------------------------------------------
    # Parsed immutable movement metadata for presentation.
    #
    # Keep JSON parsing out of Jinja. Legacy or malformed
    # metadata must not break Appliance Inventory Detail.
    # --------------------------------------------------------

    movement_meta_by_id = {}

    for movement in movements:

        meta = {}

        if movement.meta_json:

            try:
                loaded = json.loads(
                    movement.meta_json
                )

                if isinstance(
                    loaded,
                    dict,
                ):
                    meta = loaded

            except (
                TypeError,
                ValueError,
            ):
                meta = {}

        movement_meta_by_id[
            movement.id
        ] = meta

    # Categories available for descriptive correction.
    #
    # Normally only active categories are offered. If a legacy
    # appliance currently belongs to an inactive category, keep
    # that category visible so the form can still be opened and
    # corrected without silently changing it.
    categories = list(_active_categories())

    if (
        unit.category is not None
        and all(
            category.id != unit.category.id
            for category in categories
        )
    ):
        categories.append(unit.category)

        categories.sort(
            key=lambda category: (
                category.sort_order or 100,
                (category.name or "").upper(),
            )
        )

    # Internal repair executors.
    #
    # Internal repair must reference a real User whose canonical
    # application role is technician. Store the User.id in the
    # Repair Order; username is presentation/audit metadata only.
    repair_technicians = (
        User.query
        .filter(
            User.role == "technician"
        )
        .order_by(
            User.username.asc()
        )
        .all()
    )

    return render_template(
        "appliance_inventory_detail.html",
        unit=unit,
        can_pricing=can_pricing,
        can_receive=can_receive,
        movements=movements,
        movement_meta_by_id=movement_meta_by_id,
        categories=categories,
        repair_technicians=repair_technicians,
    )


# ============================================================
# Appliance Inventory - Correct descriptive / identity details
#
# Warehouse staff with appliance.receive may correct:
#   Appliance Category
#   Brand
#   Model
#   Serial
#   Size
#   Unit
#   Color
#   Condition
#   Notes
#
# Protected here:
#   AP #
#   Warehouse
#   Stock Class
#   Status
#   Pricing
#
# Operational state changes must continue through their
# dedicated Issue / Return / Repair / Disposition workflows.
#
# Every correction:
#   - updates ApplianceUnit
#   - synchronizes original ApplianceReceivingLine
#   - writes immutable ApplianceMovement audit
#   - commits once
# ============================================================

@appliance_bp.post(
    "/inventory/<int:unit_id>/details"
)
@login_required
def inventory_update_details(unit_id):

    import json
    from datetime import datetime

    from sqlalchemy import func

    from models import (
        ApplianceCategory,
        ApplianceModelSpec,
        ApplianceMovement,
        ApplianceReceivingLine,
        ApplianceUnit,
    )

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
            url_for(
                "appliance.inventory_list"
            )
        )

    # --------------------------------------------------------
    # Permission
    # --------------------------------------------------------

    if not AccessControlService.can(
        current_user,
        "appliance.receive",
        warehouse_id=unit.warehouse_id,
    ):
        flash(
            "Access denied.",
            "danger",
        )

        return redirect(
            url_for(
                "appliance.inventory_detail",
                unit_id=unit.id,
            )
        )

    detail_url = url_for(
        "appliance.inventory_detail",
        unit_id=unit.id,
    )

    # ========================================================
    # Helpers
    # ========================================================

    MODEL_SPECS_START = "[MODEL SPECS]"
    MODEL_SPECS_END = "[/MODEL SPECS]"

    def _upper_text(
        field_name,
        max_length,
    ):
        value = (
            request.form.get(field_name)
            or ""
        ).strip().upper()

        if len(value) > max_length:
            raise ValueError(
                f"{field_name} cannot exceed "
                f"{max_length} characters."
            )

        return value or None

    def _extract_model_specs_block(value):
        value = value or ""

        start_pos = value.find(
            MODEL_SPECS_START
        )

        if start_pos == -1:
            return None

        end_pos = value.find(
            MODEL_SPECS_END,
            start_pos,
        )

        if end_pos == -1:
            return None

        end_pos += len(
            MODEL_SPECS_END
        )

        block = value[
            start_pos:end_pos
        ].strip()

        return block or None

    def _remove_model_specs_block(value):
        original = (
            value
            or ""
        ).strip()

        start_pos = original.find(
            MODEL_SPECS_START
        )

        if start_pos == -1:
            return original or None

        end_pos = original.find(
            MODEL_SPECS_END,
            start_pos,
        )

        if end_pos == -1:
            return original or None

        end_pos += len(
            MODEL_SPECS_END
        )

        before = original[
            :start_pos
        ].rstrip()

        after = original[
            end_pos:
        ].lstrip()

        pieces = [
            item
            for item in (
                before,
                after,
            )
            if item
        ]

        cleaned = "\n\n".join(
            pieces
        ).strip()

        return cleaned or None

    def _merge_model_specs_block(
        original_notes,
        model_specs_block,
    ):
        """
        Preserve human/unit-specific notes.

        Replace only existing [MODEL SPECS].
        If one does not exist, append the confirmed block.
        """

        original = (
            original_notes
            or ""
        ).strip()

        block = (
            model_specs_block
            or ""
        ).strip()

        if not block:
            return original or None

        start_pos = original.find(
            MODEL_SPECS_START
        )

        if start_pos != -1:
            end_pos = original.find(
                MODEL_SPECS_END,
                start_pos,
            )

            if end_pos != -1:
                end_pos += len(
                    MODEL_SPECS_END
                )

                before = original[
                    :start_pos
                ].rstrip()

                after = original[
                    end_pos:
                ].lstrip()

                pieces = [
                    item
                    for item in (
                        before,
                        block,
                        after,
                    )
                    if item
                ]

                merged = "\n\n".join(
                    pieces
                ).strip()

                return merged or None

        if original:
            return (
                original
                + "\n\n"
                + block
            ).strip()

        return block

    # ========================================================
    # Appliance Category
    # ========================================================

    raw_category_id = (
        request.form.get("category_id")
        or ""
    ).strip()

    try:
        new_category_id = int(
            raw_category_id
        )
    except (TypeError, ValueError):
        flash(
            "Please select a valid Appliance type.",
            "danger",
        )
        return redirect(detail_url)

    new_category = db.session.get(
        ApplianceCategory,
        new_category_id,
    )

    if new_category is None:
        flash(
            "Selected Appliance type does not exist.",
            "danger",
        )
        return redirect(detail_url)

    # An inactive legacy category may remain unchanged, but
    # users cannot newly assign an inactive category.
    if (
        not new_category.is_active
        and new_category.id != unit.category_id
    ):
        flash(
            "Selected Appliance type is inactive.",
            "danger",
        )
        return redirect(detail_url)

    # ========================================================
    # Brand / Model / Serial
    # ========================================================

    try:
        new_brand = _upper_text(
            "brand",
            100,
        )

        new_model_number = _upper_text(
            "model_number",
            120,
        )

        new_serial_number = _upper_text(
            "serial_number",
            160,
        )

    except ValueError as exc:
        flash(
            str(exc),
            "danger",
        )
        return redirect(detail_url)

    # --------------------------------------------------------
    # Duplicate serial protection
    #
    # 1. another physical ApplianceUnit
    # 2. another Receiving line, including Draft receiving
    # --------------------------------------------------------

    if new_serial_number:

        duplicate_unit = (
            ApplianceUnit.query
            .filter(
                ApplianceUnit.id != unit.id,
                func.upper(
                    func.trim(
                        ApplianceUnit.serial_number
                    )
                ) == new_serial_number,
            )
            .first()
        )

        if duplicate_unit is not None:
            flash(
                "Duplicate Serial Number. "
                f"{new_serial_number} already belongs to "
                f"{duplicate_unit.inventory_number}.",
                "danger",
            )
            return redirect(detail_url)

        duplicate_line_query = (
            ApplianceReceivingLine.query
            .filter(
                func.upper(
                    func.trim(
                        ApplianceReceivingLine.serial_number
                    )
                ) == new_serial_number
            )
        )

        if unit.receiving_line_id:
            duplicate_line_query = (
                duplicate_line_query
                .filter(
                    ApplianceReceivingLine.id
                    != unit.receiving_line_id
                )
            )

        duplicate_line = (
            duplicate_line_query.first()
        )

        if duplicate_line is not None:
            duplicate_line_unit = (
                ApplianceUnit.query
                .filter(
                    ApplianceUnit.receiving_line_id
                    == duplicate_line.id
                )
                .first()
            )

            if (
                duplicate_line_unit is None
                or duplicate_line_unit.id != unit.id
            ):
                flash(
                    "Duplicate Serial Number. "
                    f"{new_serial_number} already exists "
                    "in Appliance Receiving.",
                    "danger",
                )
                return redirect(detail_url)

    # ========================================================
    # Size
    # ========================================================

    raw_size = (
        request.form.get("size_value")
        or ""
    ).strip()

    new_size_value = None

    if raw_size:
        try:
            new_size_value = float(
                raw_size
            )
        except (TypeError, ValueError):
            flash(
                "Size must be a valid number.",
                "danger",
            )
            return redirect(detail_url)

        if new_size_value < 0:
            flash(
                "Size cannot be negative.",
                "danger",
            )
            return redirect(detail_url)

    # ========================================================
    # Unit / Color
    # ========================================================

    new_size_unit = (
        request.form.get("size_unit")
        or ""
    ).strip().upper()

    if len(new_size_unit) > 20:
        flash(
            "Unit cannot exceed 20 characters.",
            "danger",
        )
        return redirect(detail_url)

    new_size_unit = (
        new_size_unit or None
    )

    new_color = (
        request.form.get("color")
        or ""
    ).strip().upper()

    if len(new_color) > 80:
        flash(
            "Color cannot exceed 80 characters.",
            "danger",
        )
        return redirect(detail_url)

    new_color = (
        new_color or None
    )

    # ========================================================
    # Condition
    #
    # Keep the same warehouse values already used by Receiving,
    # plus REPAIRED for the repair lifecycle.
    # ========================================================

    new_condition = (
        request.form.get("condition")
        or "new"
    ).strip().lower()

    allowed_conditions = {
        "new",
        "used",
        "open_box",
        "damaged",
        "repaired",
    }

    if new_condition not in allowed_conditions:
        flash(
            "Invalid appliance Condition.",
            "danger",
        )
        return redirect(detail_url)

    # ========================================================
    # Notes
    # ========================================================

    new_notes = (
        request.form.get("notes")
        or ""
    ).strip()

    if len(new_notes) > 5000:
        flash(
            "Notes cannot exceed 5000 characters.",
            "danger",
        )
        return redirect(detail_url)

    new_notes = (
        new_notes or None
    )

    # ========================================================
    # Snapshot OLD identity first
    # ========================================================

    old_category_id = unit.category_id
    old_brand = unit.brand
    old_model_number = unit.model_number
    old_serial_number = unit.serial_number

    identity_changed = any(
        (
            old_category_id
            != new_category_id,

            (old_brand or "")
            != (new_brand or ""),

            (old_model_number or "")
            != (new_model_number or ""),

            (old_serial_number or "")
            != (new_serial_number or ""),
        )
    )

    model_identity_changed = any(
        (
            (old_brand or "")
            != (new_brand or ""),

            (old_model_number or "")
            != (new_model_number or ""),
        )
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # If Brand or Model was corrected, old MODEL SPECS may
    # belong to the old model. Never silently attach those
    # specifications to a different Brand+Model.
    #
    # Human notes are preserved; only the structured specs
    # block is removed. User may SAVE identity, then run
    # FIND SPECS for the corrected model.
    # --------------------------------------------------------

    removed_stale_specs = False

    if (
        model_identity_changed
        and _extract_model_specs_block(
            new_notes
        )
    ):
        new_notes = (
            _remove_model_specs_block(
                new_notes
            )
        )

        removed_stale_specs = True

    # ========================================================
    # OLD / NEW audit snapshots
    # ========================================================

    old_values = {
        "category_id":
            unit.category_id,

        "category":
            (
                unit.category.name
                if unit.category is not None
                else None
            ),

        "brand":
            unit.brand,

        "model_number":
            unit.model_number,

        "serial_number":
            unit.serial_number,

        "size_value":
            unit.size_value,

        "size_unit":
            unit.size_unit,

        "color":
            unit.color,

        "condition":
            unit.condition,

        "notes":
            unit.notes,
    }

    new_values = {
        "category_id":
            new_category.id,

        "category":
            new_category.name,

        "brand":
            new_brand,

        "model_number":
            new_model_number,

        "serial_number":
            new_serial_number,

        "size_value":
            new_size_value,

        "size_unit":
            new_size_unit,

        "color":
            new_color,

        "condition":
            new_condition,

        "notes":
            new_notes,
    }

    if old_values == new_values:
        flash(
            "No changes were made.",
            "info",
        )
        return redirect(detail_url)

    now = datetime.utcnow()

    try:

        # ====================================================
        # 1. Physical ApplianceUnit = current source of truth
        # ====================================================

        unit.category_id = (
            new_category.id
        )

        unit.brand = (
            new_brand
        )

        unit.model_number = (
            new_model_number
        )

        unit.serial_number = (
            new_serial_number
        )

        unit.size_value = (
            new_size_value
        )

        unit.size_unit = (
            new_size_unit
        )

        unit.color = (
            new_color
        )

        unit.condition = (
            new_condition
        )

        unit.notes = (
            new_notes
        )

        unit.updated_at = now
        unit.updated_by_id = (
            current_user.id
        )

        # ====================================================
        # 2. Keep original Receiving lineage synchronized
        # ====================================================

        receiving_line = None

        if unit.receiving_line_id:
            receiving_line = db.session.get(
                ApplianceReceivingLine,
                unit.receiving_line_id,
            )

        if receiving_line is not None:

            receiving_line.category_id = (
                new_category.id
            )

            receiving_line.brand = (
                new_brand
            )

            receiving_line.model_number = (
                new_model_number
            )

            receiving_line.serial_number = (
                new_serial_number
            )

            receiving_line.size_value = (
                new_size_value
            )

            receiving_line.size_unit = (
                new_size_unit
            )

            receiving_line.color = (
                new_color
            )

            receiving_line.condition = (
                new_condition
            )

            receiving_line.notes = (
                new_notes
            )

            receiving_line.updated_at = now
            receiving_line.updated_by_id = (
                current_user.id
            )

            receiving = (
                receiving_line.receiving
            )

            if receiving is not None:
                receiving.updated_at = now
                receiving.updated_by_id = (
                    current_user.id
                )

        # ====================================================
        # 3. Immutable audit entry
        # ====================================================

        movement_type = (
            "IDENTITY_CORRECTION"
            if identity_changed
            else "DETAILS_UPDATE"
        )

        reason_code = (
            "IDENTITY_CORRECTED"
            if identity_changed
            else "DETAILS_UPDATED"
        )

        movement_notes = (
            "Appliance identity/descriptive data corrected "
            "from Inventory."
            if identity_changed
            else
            "Appliance descriptive details updated "
            "from Inventory."
        )

        movement = ApplianceMovement(
            appliance_unit_id=unit.id,

            movement_type=movement_type,

            from_warehouse_id=(
                unit.warehouse_id
            ),

            to_warehouse_id=(
                unit.warehouse_id
            ),

            reason_code=reason_code,

            notes=movement_notes,

            meta_json=json.dumps(
                {
                    "inventory_number":
                        unit.inventory_number,

                    "source":
                        "inventory_detail",

                    "identity_changed":
                        identity_changed,

                    "model_identity_changed":
                        model_identity_changed,

                    "stale_model_specs_removed":
                        removed_stale_specs,

                    "old":
                        old_values,

                    "new":
                        new_values,

                    "receiving_line_id":
                        unit.receiving_line_id,
                },
                ensure_ascii=False,
            ),

            actor_id=current_user.id,
            created_at=now,
        )

        db.session.add(
            movement
        )

        # ====================================================
        # 4. MODEL SPECS CATALOG + PROPAGATION
        #
        # Only use a confirmed [MODEL SPECS] block when Brand
        # and Model themselves were NOT changed in this SAVE.
        #
        # If Brand/Model was corrected:
        #   SAVE identity first
        #   then FIND SPECS
        #   then APPLY + SAVE
        # ====================================================

        specs_block = (
            _extract_model_specs_block(
                new_notes
            )
        )

        brand_key = " ".join(
            (
                unit.brand
                or ""
            ).strip().upper().split()
        )

        model_key = " ".join(
            (
                unit.model_number
                or ""
            ).strip().upper().split()
        )

        propagated_count = 0

        if (
            specs_block
            and brand_key
            and model_key
            and not model_identity_changed
        ):

            # ----------------------------------------------
            # UPSERT confirmed Model Specs Catalog
            # ----------------------------------------------

            model_spec = (
                ApplianceModelSpec.query
                .filter(
                    ApplianceModelSpec.normalized_brand
                    == brand_key,

                    ApplianceModelSpec.normalized_model
                    == model_key,
                )
                .first()
            )

            if model_spec is None:
                model_spec = (
                    ApplianceModelSpec(
                        brand=(
                            unit.brand
                            or brand_key
                        ).strip(),

                        model_number=(
                            unit.model_number
                            or model_key
                        ).strip(),

                        normalized_brand=
                            brand_key,

                        normalized_model=
                            model_key,

                        created_at=now,
                    )
                )

                db.session.add(
                    model_spec
                )

            model_spec.brand = (
                unit.brand
                or brand_key
            ).strip()

            model_spec.model_number = (
                unit.model_number
                or model_key
            ).strip()

            model_spec.normalized_brand = (
                brand_key
            )

            model_spec.normalized_model = (
                model_key
            )

            model_spec.size_value = (
                new_size_value
            )

            model_spec.size_unit = (
                new_size_unit
            )

            model_spec.color = (
                new_color
            )

            model_spec.notes_block = (
                specs_block
            )

            model_spec.exact_model_confirmed = (
                True
            )

            model_spec.confirmed_model = (
                unit.model_number
            )

            model_spec.confidence = (
                "confirmed"
            )

            if not model_spec.source_name:
                model_spec.source_name = (
                    "LOCAL MODEL CATALOG"
                )

            model_spec.confirmed_by_id = (
                current_user.id
            )

            model_spec.confirmed_at = (
                now
            )

            model_spec.updated_at = (
                now
            )

            # ----------------------------------------------
            # Find existing physical units with same
            # normalized Brand + Model.
            # ----------------------------------------------

            matching_units = (
                ApplianceUnit.query
                .filter(
                    func.upper(
                        func.trim(
                            ApplianceUnit.brand
                        )
                    ) == brand_key,

                    func.upper(
                        func.trim(
                            ApplianceUnit.model_number
                        )
                    ) == model_key,
                )
                .all()
            )

            for other_unit in matching_units:

                if other_unit.id == unit.id:
                    continue

                old_other = {
                    "size_value":
                        other_unit.size_value,

                    "size_unit":
                        other_unit.size_unit,

                    "color":
                        other_unit.color,

                    "notes":
                        other_unit.notes,
                }

                merged_notes = (
                    _merge_model_specs_block(
                        other_unit.notes,
                        specs_block,
                    )
                )

                other_unit.size_value = (
                    new_size_value
                )

                other_unit.size_unit = (
                    new_size_unit
                )

                other_unit.color = (
                    new_color
                )

                other_unit.notes = (
                    merged_notes
                )

                other_unit.updated_at = now
                other_unit.updated_by_id = (
                    current_user.id
                )

                other_receiving_line = None

                if other_unit.receiving_line_id:
                    other_receiving_line = (
                        db.session.get(
                            ApplianceReceivingLine,
                            other_unit.receiving_line_id,
                        )
                    )

                if other_receiving_line is not None:

                    other_receiving_line.size_value = (
                        new_size_value
                    )

                    other_receiving_line.size_unit = (
                        new_size_unit
                    )

                    other_receiving_line.color = (
                        new_color
                    )

                    other_receiving_line.notes = (
                        merged_notes
                    )

                    other_receiving_line.updated_at = (
                        now
                    )

                    other_receiving_line.updated_by_id = (
                        current_user.id
                    )

                    if (
                        other_receiving_line.receiving
                        is not None
                    ):
                        other_receiving_line.receiving.updated_at = (
                            now
                        )

                        other_receiving_line.receiving.updated_by_id = (
                            current_user.id
                        )

                new_other = {
                    "size_value":
                        other_unit.size_value,

                    "size_unit":
                        other_unit.size_unit,

                    "color":
                        other_unit.color,

                    "notes":
                        other_unit.notes,
                }

                if old_other != new_other:

                    propagation_movement = (
                        ApplianceMovement(
                            appliance_unit_id=
                                other_unit.id,

                            movement_type=
                                "MODEL_SPECS_SYNC",

                            from_warehouse_id=
                                other_unit.warehouse_id,

                            to_warehouse_id=
                                other_unit.warehouse_id,

                            reason_code=
                                "MODEL_SPECS_PROPAGATED",

                            notes=(
                                "Confirmed model specifications "
                                "propagated from local Model "
                                "Specs Catalog."
                            ),

                            meta_json=json.dumps(
                                {
                                    "source":
                                        "appliance_model_spec",

                                    "normalized_brand":
                                        brand_key,

                                    "normalized_model":
                                        model_key,

                                    "old":
                                        old_other,

                                    "new":
                                        new_other,

                                    "confirmed_from_unit_id":
                                        unit.id,

                                    "confirmed_from_inventory_number":
                                        unit.inventory_number,
                                },
                                ensure_ascii=False,
                            ),

                            actor_id=
                                current_user.id,

                            created_at=
                                now,
                        )
                    )

                    db.session.add(
                        propagation_movement
                    )

                    propagated_count += 1

        # ====================================================
        # ONE COMMIT
        # ====================================================

        db.session.commit()

        message = (
            f"Details saved for "
            f"{unit.inventory_number}."
        )

        if removed_stale_specs:
            message += (
                " Brand/Model changed, so the old Model Specs "
                "block was removed. Run FIND SPECS for the "
                "corrected model."
            )

        flash(
            message,
            "success",
        )

    except Exception:
        db.session.rollback()
        raise

    return redirect(
        url_for(
            "appliance.inventory_detail",
            unit_id=unit.id,
        )
    )


# ============================================================
# Appliance Inventory - AI Model Specs Preview
#
# IMPORTANT:
#   Research only.
#   NO database writes.
#   NO commit.
#   User must explicitly SAVE the normal Edit Details form.
# ============================================================

@appliance_bp.post(
    "/inventory/<int:unit_id>/find-specs"
)
@login_required
def inventory_find_specs(unit_id):

    from flask import current_app, jsonify

    from appliance.services.model_specs_service import (
        ModelSpecsService,
    )

    unit = db.session.get(
        ApplianceUnit,
        unit_id,
    )

    if unit is None:
        return jsonify(
            {
                "ok": False,
                "error": "Appliance unit not found.",
            }
        ), 404

    if not AccessControlService.can(
        current_user,
        "appliance.receive",
        warehouse_id=unit.warehouse_id,
    ):
        return jsonify(
            {
                "ok": False,
                "error": "Access denied.",
            }
        ), 403

    brand = (
        unit.brand
        or ""
    ).strip()

    model = (
        unit.model_number
        or ""
    ).strip()

    serial = (
        unit.serial_number
        or ""
    ).strip() or None

    if not brand:
        return jsonify(
            {
                "ok": False,
                "error": (
                    "Brand is missing. "
                    "FIND SPECS requires Brand + Model."
                ),
            }
        ), 400

    if not model:
        return jsonify(
            {
                "ok": False,
                "error": (
                    "Model is missing. "
                    "FIND SPECS requires Brand + Model."
                ),
            }
        ), 400

    # ========================================================
    # LOCAL MODEL SPECS CACHE LOOKUP
    #
    # Confirmed Brand + Model?
    #   YES -> return local result immediately.
    #          NO OpenAI call.
    #          NO web search.
    #
    #   NO  -> continue to ModelSpecsService below.
    # ========================================================

    import json

    from models import ApplianceModelSpec


    brand_key = " ".join(
        brand.upper().split()
    )

    model_key = " ".join(
        model.upper().split()
    )


    cached = (
        ApplianceModelSpec.query
        .filter(
            ApplianceModelSpec.normalized_brand
            == brand_key,

            ApplianceModelSpec.normalized_model
            == model_key,

            ApplianceModelSpec.confirmed_at
            .isnot(None),
        )
        .first()
    )


    if cached is not None:

        sources = []

        if cached.sources_json:

            try:
                parsed_sources = json.loads(
                    cached.sources_json
                )

                if isinstance(
                    parsed_sources,
                    list,
                ):
                    sources = (
                        parsed_sources
                    )

            except Exception:
                sources = []


        data = {
            "exact_model_confirmed":
                bool(
                    cached.exact_model_confirmed
                ),

            "confirmed_model":
                cached.confirmed_model
                or cached.model_number,

            "suggested_model":
                cached.suggested_model,

            "match_notes":
                cached.match_notes
                or (
                    "Loaded from confirmed "
                    "local Model Specs Catalog."
                ),

            "appliance_type":
                cached.appliance_type,

            "size_value":
                cached.size_value,

            "size_unit":
                cached.size_unit,

            "color":
                cached.color,

            "notes_block":
                cached.notes_block,

            "source_name":
                cached.source_name
                or "LOCAL MODEL CATALOG",

            "source_url":
                cached.source_url,

            "confidence":
                cached.confidence
                or "confirmed",

            "sources":
                sources,

            # Extra fields are safe if current JS ignores them.
            "cache_hit":
                True,

            "lookup_source":
                "LOCAL_CACHE",
        }


        current_app.logger.info(
            "MODEL_SPECS_CACHE_HIT "
            "unit_id=%s brand=%s model=%s cache_id=%s",
            unit.id,
            brand,
            model,
            cached.id,
        )


        return jsonify(
            {
                "ok": True,
                "result": data,
            }
        )


    try:

        service = ModelSpecsService()

        result = service.find_specs(
            brand=brand,
            model=model,
            serial=serial,
        )

        data = result.to_dict()

        # ====================================================
        # REMEMBER CONFIRMED MODEL SPECS
        #
        # FIND SPECS remains read-only for ApplianceUnit.
        # We only persist reusable Brand + Model knowledge
        # into ApplianceModelSpec.
        #
        # Next lookup for the same exact Brand + Model will
        # use LOCAL CACHE and will not call OpenAI/Web.
        # ====================================================

        if bool(
            data.get("exact_model_confirmed")
        ):

            cached_model = (
                ApplianceModelSpec.query
                .filter(
                    ApplianceModelSpec.normalized_brand
                    == brand_key,

                    ApplianceModelSpec.normalized_model
                    == model_key,
                )
                .first()
            )

            if cached_model is None:

                cached_model = ApplianceModelSpec(
                    brand=brand,
                    model_number=model,
                    normalized_brand=brand_key,
                    normalized_model=model_key,
                )

                db.session.add(
                    cached_model
                )

            cached_model.brand = brand
            cached_model.model_number = model

            cached_model.normalized_brand = (
                brand_key
            )

            cached_model.normalized_model = (
                model_key
            )

            cached_model.size_value = (
                data.get("size_value")
            )

            cached_model.size_unit = (
                data.get("size_unit")
            )

            cached_model.color = (
                data.get("color")
            )

            cached_model.notes_block = (
                data.get("notes_block")
            )

            cached_model.exact_model_confirmed = True

            cached_model.confirmed_model = (
                data.get("confirmed_model")
                or model
            )

            cached_model.suggested_model = (
                data.get("suggested_model")
            )

            cached_model.appliance_type = (
                data.get("appliance_type")
            )

            cached_model.match_notes = (
                data.get("match_notes")
            )

            cached_model.confidence = (
                data.get("confidence")
                or "confirmed"
            )

            cached_model.source_name = (
                data.get("source_name")
            )

            cached_model.source_url = (
                data.get("source_url")
            )

            cached_model.sources_json = (
                json.dumps(
                    data.get("sources") or [],
                    ensure_ascii=False,
                )
            )

            cached_model.confirmed_by_id = (
                current_user.id
            )

            cached_model.confirmed_at = (
                datetime.utcnow()
            )

            db.session.commit()

            current_app.logger.info(
                "MODEL_SPECS_CACHE_SAVED "
                "unit_id=%s brand=%s model=%s "
                "cache_id=%s",
                unit.id,
                brand,
                model,
                cached_model.id,
            )

            # Inform frontend/debugging that this result
            # has now been persisted in local catalog.
            data["cache_saved"] = True
            data["cache_id"] = cached_model.id

        return jsonify(
            {
                "ok": True,
                "result": data,
            }
        )

    except Exception as exc:

        current_app.logger.exception(
            "MODEL_SPECS_LOOKUP_FAILED "
            "unit_id=%s brand=%s model=%s",
            unit.id,
            brand,
            model,
        )

        return jsonify(
            {
                "ok": False,
                "error": (
                    "Model specs lookup failed. "
                    f"{exc}"
                ),
            }
        ), 500


# ============================================================
# Appliance Inventory disposition/status action
# ============================================================

@appliance_bp.post(
    "/inventory/<int:unit_id>/disposition"
)
@login_required
def inventory_disposition(unit_id):

    action = (
        request.form.get("action")
        or ""
    ).strip().upper()

    extra_movement_meta = None

    if action == "SEND_TO_REPAIR":

        extra_movement_meta = {
            "repair_type": (
                request.form.get(
                    "repair_type"
                )
                or ""
            ).strip().lower(),

            "repair_vendor": (
                request.form.get(
                    "repair_vendor"
                )
                or ""
            ).strip(),

            "repair_technician_id": (
                request.form.get(
                    "repair_technician_id"
                )
                or ""
            ).strip(),
        }

    try:

        unit = (
            ApplianceIssueService
            .change_inventory_disposition(
                actor=current_user,
                appliance_unit_id=unit_id,
                action=action,
                reason_code=(
                    request.form.get("reason_code")
                    or None
                ),
                notes=(
                    request.form.get("notes")
                    or None
                ),
                extra_movement_meta=(
                    extra_movement_meta
                ),
            )
        )

        flash(
            f"{unit.inventory_number} status changed to "
            f"{(unit.status or '').replace('_', ' ').upper()}.",
            "success",
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

    return redirect(
        url_for(
            "appliance.inventory_detail",
            unit_id=unit_id,
        )
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

                "stock_class": (
                    unit.stock_class
                    or "new"
                ).replace(
                    "_",
                    " ",
                ).upper(),

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

    # --------------------------------------------------------
    # Replacement choices per active Issue line.
    #
    # Only:
    #   AVAILABLE
    #   same warehouse
    #   same appliance category/type
    # --------------------------------------------------------

    replacement_choices = {}

    for line in issue.lines:

        if (
            line.status
            or ""
        ).strip().lower() != "issued":
            continue

        unit = line.appliance_unit

        if unit is None:
            continue

        choices = (
            ApplianceUnit.query
            .filter(
                ApplianceUnit.warehouse_id
                == issue.warehouse_id,

                ApplianceUnit.category_id
                == unit.category_id,

                ApplianceUnit.status
                == "available",
            )
            .order_by(
                ApplianceUnit.brand.asc(),
                ApplianceUnit.model_number.asc(),
                ApplianceUnit.serial_number.asc(),
                ApplianceUnit.inventory_number.asc(),
            )
            .all()
        )

        replacement_choices[
            line.id
        ] = choices

    return render_template(
        "appliance_issue_detail.html",
        issue=issue,
        auto_print=auto_print,
        replacement_choices=replacement_choices,
    )




# ============================================================
# REPLACE ONE ISSUED APPLIANCE
# ============================================================

@appliance_bp.post(
    "/issues/lines/<int:line_id>/replace"
)
@login_required
def issue_replace_line(line_id):

    from models import ApplianceIssueLine

    line = db.session.get(
        ApplianceIssueLine,
        line_id,
    )

    if line is None:

        flash(
            "Appliance Issue line not found.",
            "danger",
        )

        return redirect(
            url_for(
                "appliance.inventory_list"
            )
        )

    issue_id = line.issue_id

    try:

        try:
            new_unit_id = int(
                request.form.get(
                    "new_appliance_unit_id"
                )
                or 0
            )
        except (
            TypeError,
            ValueError,
        ):
            new_unit_id = 0

        if new_unit_id <= 0:
            raise ApplianceIssueError(
                "Select replacement appliance."
            )

        result = (
            ApplianceIssueService
            .replace_appliance(
                actor=current_user,
                issue_line_id=line_id,
                new_appliance_unit_id=new_unit_id,
                notes=(
                    request.form.get("notes")
                    or None
                ),
            )
        )

        flash(
            f"{result['old_inventory_number']} replaced with "
            f"{result['new_inventory_number']}.",
            "success",
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

    return redirect(
        url_for(
            "appliance.issue_detail",
            issue_id=issue_id,
        )
    )


# ============================================================
# RETURN ONE APPLIANCE TO STOCK
# ============================================================

@appliance_bp.post(
    "/issues/lines/<int:line_id>/return"
)
@login_required
def issue_return_line(line_id):

    from models import ApplianceIssueLine

    line = db.session.get(
        ApplianceIssueLine,
        line_id,
    )

    if line is None:

        flash(
            "Appliance Issue line not found.",
            "danger",
        )

        return redirect(
            url_for(
                "appliance.inventory_list"
            )
        )

    issue_id = line.issue_id

    try:

        ApplianceIssueService.return_to_stock(
            actor=current_user,
            issue_line_id=line_id,
            notes=(
                request.form.get("notes")
                or None
            ),
        )

        flash(
            f"{line.inventory_number_snapshot} "
            "returned to stock.",
            "success",
        )

    except ApplianceIssueError as exc:

        db.session.rollback()

        flash(
            str(exc),
            "danger",
        )

    return redirect(
        url_for(
            "appliance.issue_detail",
            issue_id=issue_id,
        )
    )


# ============================================================
# REMOVE ONE APPLIANCE FROM ISSUE
# SUPERADMIN correction only
# ============================================================

@appliance_bp.post(
    "/issues/lines/<int:line_id>/remove"
)
@login_required
def issue_remove_line(line_id):

    from models import ApplianceIssueLine

    try:

        result = (
            ApplianceIssueService
            .remove_line_from_issue(
                actor=current_user,
                issue_line_id=line_id,
            )
        )

        flash(
            f"{result['inventory_number']} removed from "
            f"{result['issue_number']}.",
            "success",
        )

        if result["issue_deleted"]:

            flash(
                f"{result['issue_number']} had no remaining "
                "appliances and was deleted.",
                "success",
            )

            return redirect(
                url_for(
                    "appliance.inventory_list"
                )
            )

        return redirect(
            url_for(
                "appliance.issue_detail",
                issue_id=result["issue_id"],
            )
        )

    except ApplianceIssueError as exc:

        db.session.rollback()

        flash(
            str(exc),
            "danger",
        )

        line = db.session.get(
            ApplianceIssueLine,
            line_id,
        )

        if line is not None:

            return redirect(
                url_for(
                    "appliance.issue_detail",
                    issue_id=line.issue_id,
                )
            )

        return redirect(
            url_for(
                "appliance.inventory_list"
            )
        )

# ============================================================
# DELETE APPLIANCE ISSUE
#
# Superadmin-only correction tool.
#
# Business rules live in ApplianceIssueService.
# This route only handles HTTP / flash / redirect.
# ============================================================

@appliance_bp.post(
    "/issues/<int:issue_id>/delete"
)
@login_required
def issue_delete(issue_id):

    # Defense-in-depth:
    # physical Issue deletion is a SUPERADMIN-only
    # correction operation. The service enforces the same rule.
    role = (
        getattr(current_user, "role", "")
        or ""
    ).strip().lower()

    if role != "superadmin":
        flash(
            "Only SUPERADMIN can delete "
            "an Appliance Issue.",
            "danger",
        )

        return redirect(
            url_for(
                "appliance.inventory_list"
            )
        )

    try:

        result = (
            ApplianceIssueService
            .delete_issue_correction(
                actor=current_user,
                issue_id=issue_id,
            )
        )

        flash(
            f"{result['issue_number']} deleted. "
            f"{result['deleted_units']} appliance(s) "
            "returned to AVAILABLE.",
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


# ============================================================
# APPLIANCE REPAIR ORDERS
# Read-only Repair control center.
# ============================================================

@appliance_bp.get("/repair-orders")
@login_required
def repair_orders_registry():
    """
    Read-only Repair Order registry.

    Access is warehouse-scoped through appliance.view.
    No Repair lifecycle mutation is performed here.
    """

    from datetime import datetime

    from models import ApplianceRepairOrder

    allowed_warehouse_ids = _warehouse_ids_for(
        "appliance.view"
    )

    status = (
        request.args.get("status")
        or "open"
    ).strip().lower()

    if status not in (
        "all",
        "open",
        "completed",
        "vendor_return",
        "scrapped",
        "cancelled",
    ):
        status = "open"

    q = (
        request.args.get("q")
        or ""
    ).strip()

    if not allowed_warehouse_ids:
        return render_template(
            "appliance_repair_orders.html",
            repair_orders=[],
            q=q,
            status=status,
            open_count=0,
        )

    query = (
        ApplianceRepairOrder.query
        .filter(
            ApplianceRepairOrder.warehouse_id.in_(
                allowed_warehouse_ids
            )
        )
    )

    if status != "all":
        query = query.filter(
            ApplianceRepairOrder.status == status
        )

    if q:
        like = f"%{q}%"

        query = query.filter(
            db.or_(
                ApplianceRepairOrder.repair_number.ilike(
                    like
                ),
                ApplianceRepairOrder.repair_vendor.ilike(
                    like
                ),
                ApplianceRepairOrder.provider_reference.ilike(
                    like
                ),
            )
        )

    repair_orders = (
        query
        .order_by(
            ApplianceRepairOrder.sent_at.desc(),
            ApplianceRepairOrder.id.desc(),
        )
        .all()
    )

    open_count = (
        ApplianceRepairOrder.query
        .filter(
            ApplianceRepairOrder.warehouse_id.in_(
                allowed_warehouse_ids
            ),
            ApplianceRepairOrder.status == "open",
        )
        .count()
    )

    now = datetime.utcnow()

    for repair in repair_orders:
        if (
            repair.status == "open"
            and repair.sent_at is not None
        ):
            delta = now - repair.sent_at
            repair.days_in_repair = max(
                0,
                delta.days,
            )
        else:
            repair.days_in_repair = None

    return render_template(
        "appliance_repair_orders.html",
        repair_orders=repair_orders,
        q=q,
        status=status,
        open_count=open_count,
    )


# ============================================================
# APPLIANCE REPAIR ORDER DETAIL
# Read-only.
# ============================================================

@appliance_bp.get("/repair-orders/<int:repair_id>")
@login_required
def repair_order_detail(repair_id):
    """
    Read-only detail for one Repair Order.

    Access is warehouse-scoped through appliance.view.
    No Repair lifecycle mutation is performed here.
    """

    from models import ApplianceRepairOrder

    repair = db.session.get(
        ApplianceRepairOrder,
        repair_id,
    )

    if repair is None:
        flash(
            "Repair Order not found.",
            "danger",
        )
        return redirect(
            url_for(
                "appliance.repair_orders_registry"
            )
        )

    if not AccessControlService.can(
        current_user,
        "appliance.view",
        warehouse_id=repair.warehouse_id,
    ):
        flash(
            "Access denied.",
            "danger",
        )
        return redirect(
            url_for(
                "appliance.repair_orders_registry"
            )
        )

    return render_template(
        "appliance_repair_order_detail.html",
        repair=repair,
    )


# ============================================================
# APPLIANCE REPAIR ORDER LIFECYCLE ACTION
# ============================================================

@appliance_bp.post(
    "/repair-orders/<int:repair_id>/action"
)
@login_required
def repair_order_action(repair_id):
    """
    Execute a lifecycle outcome for one OPEN Repair Order.

    Business rules and atomic mutation remain inside
    ApplianceIssueService.change_inventory_disposition().
    """
    from models import ApplianceRepairOrder

    from services.appliance_issue_service import (
        ApplianceIssueAccessDenied,
        ApplianceIssueError,
        ApplianceIssueService,
    )

    repair = db.session.get(
        ApplianceRepairOrder,
        repair_id,
    )

    if repair is None:
        flash(
            "Repair Order not found.",
            "danger",
        )
        return redirect(
            url_for(
                "appliance.repair_orders_registry"
            )
        )

    if not AccessControlService.can(
        current_user,
        "appliance.view",
        warehouse_id=repair.warehouse_id,
    ):
        flash(
            "Access denied.",
            "danger",
        )
        return redirect(
            url_for(
                "appliance.repair_orders_registry"
            )
        )

    if repair.status != "open":
        flash(
            f"{repair.repair_number} is not OPEN.",
            "danger",
        )
        return redirect(
            url_for(
                "appliance.repair_order_detail",
                repair_id=repair.id,
            )
        )

    action = (
        request.form.get("action")
        or ""
    ).strip().upper()

    allowed_actions = {
        "RETURN_FROM_REPAIR_TO_NEW",
        "RETURN_FROM_REPAIR_TO_LOANER",
        "REPAIR_TO_VENDOR_RETURN",
        "REPAIR_TO_SCRAP",
    }

    if action not in allowed_actions:
        flash(
            "Unsupported Repair Order action.",
            "danger",
        )
        return redirect(
            url_for(
                "appliance.repair_order_detail",
                repair_id=repair.id,
            )
        )

    try:
        ApplianceIssueService.change_inventory_disposition(
            actor=current_user,
            appliance_unit_id=repair.appliance_unit_id,
            action=action,
        )

        flash(
            f"{repair.repair_number} updated successfully.",
            "success",
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

    return redirect(
        url_for(
            "appliance.repair_order_detail",
            repair_id=repair.id,
        )
    )


# ============================================================
# APPLIANCE VENDOR RETURN CONTROL
# ============================================================

@appliance_bp.get("/vendor-returns")
@login_required
def vendor_returns_queue():
    """
    Manager control screen.

    PENDING:
        Physical appliance is still under company control but
        cannot be issued.

    HISTORY:
        Physical return to supplier was confirmed.

    Vendor is derived from original Appliance Receiving.
    RMA/reference is stored in the final movement meta_json.
    """

    import json

    from models import (
        ApplianceMovement,
        ApplianceUnit,
    )

    allowed_warehouse_ids = _warehouse_ids_for(
        "appliance.view"
    )

    if not allowed_warehouse_ids:

        return render_template(
            "appliance_vendor_returns.html",
            pending_rows=[],
            history_rows=[],
            pending_count=0,
            history_count=0,
        )

    # --------------------------------------------------------
    # Pending physical units
    # --------------------------------------------------------

    pending_units = (
        ApplianceUnit.query
        .filter(
            ApplianceUnit.warehouse_id.in_(
                allowed_warehouse_ids
            ),
            ApplianceUnit.status
            == "vendor_return_pending",
        )
        .order_by(
            ApplianceUnit.updated_at.asc(),
            ApplianceUnit.id.asc(),
        )
        .all()
    )

    pending_ids = [
        unit.id
        for unit in pending_units
    ]

    pending_movement_map = {}

    if pending_ids:

        pending_movements = (
            ApplianceMovement.query
            .filter(
                ApplianceMovement.appliance_unit_id.in_(
                    pending_ids
                ),
                ApplianceMovement.movement_type
                == "VENDOR_RETURN_PENDING",
            )
            .order_by(
                ApplianceMovement.appliance_unit_id.asc(),
                ApplianceMovement.created_at.desc(),
                ApplianceMovement.id.desc(),
            )
            .all()
        )

        for movement in pending_movements:

            if (
                movement.appliance_unit_id
                not in pending_movement_map
            ):
                pending_movement_map[
                    movement.appliance_unit_id
                ] = movement

    pending_rows = []

    for unit in pending_units:

        movement = pending_movement_map.get(
            unit.id
        )

        receiving = None

        if (
            unit.receiving_line
            and unit.receiving_line.receiving
        ):
            receiving = (
                unit.receiving_line.receiving
            )

        vendor_name = (
            receiving.supplier_name
            if receiving
            else ""
        ) or ""

        receiving_number = (
            receiving.receiving_number
            if receiving
            else ""
        ) or ""

        purchase_invoice = (
            receiving.invoice_number
            if receiving
            else ""
        ) or ""

        can_confirm = (
            AccessControlService.can(
                current_user,
                "appliance.vendor_return",
                warehouse_id=unit.warehouse_id,
            )
        )

        pending_rows.append(
            {
                "unit": unit,
                "movement": movement,
                "vendor_name": vendor_name,
                "receiving_number":
                    receiving_number,
                "purchase_invoice":
                    purchase_invoice,
                "can_confirm":
                    can_confirm,
            }
        )

    # --------------------------------------------------------
    # Confirmed Vendor Return History
    # --------------------------------------------------------

    history_movements = (
        ApplianceMovement.query
        .join(
            ApplianceUnit,
            ApplianceUnit.id
            == ApplianceMovement.appliance_unit_id,
        )
        .filter(
            ApplianceUnit.warehouse_id.in_(
                allowed_warehouse_ids
            ),
            ApplianceMovement.movement_type
            == "VENDOR_RETURN",
        )
        .order_by(
            ApplianceMovement.created_at.desc(),
            ApplianceMovement.id.desc(),
        )
        .limit(500)
        .all()
    )

    history_rows = []

    for movement in history_movements:

        unit = movement.appliance_unit

        receiving = None

        if (
            unit
            and unit.receiving_line
            and unit.receiving_line.receiving
        ):
            receiving = (
                unit.receiving_line.receiving
            )

        meta = {}

        if movement.meta_json:

            try:
                loaded = json.loads(
                    movement.meta_json
                )

                if isinstance(
                    loaded,
                    dict,
                ):
                    meta = loaded

            except Exception:
                meta = {}

        vendor_name = (
            meta.get("vendor_name")
            or (
                receiving.supplier_name
                if receiving
                else ""
            )
            or ""
        )

        rma_reference = (
            meta.get("rma_reference")
            or ""
        )

        history_rows.append(
            {
                "movement": movement,
                "unit": unit,
                "vendor_name": vendor_name,
                "rma_reference":
                    rma_reference,
                "purchase_invoice":
                    (
                        receiving.invoice_number
                        if receiving
                        else ""
                    )
                    or "",
            }
        )

    return render_template(
        "appliance_vendor_returns.html",
        pending_rows=pending_rows,
        history_rows=history_rows,
        pending_count=len(
            pending_rows
        ),
        history_count=len(
            history_rows
        ),
    )


# ============================================================
# CONFIRM PHYSICAL VENDOR RETURN
# ============================================================

@appliance_bp.post(
    "/vendor-returns/<int:unit_id>/confirm"
)
@login_required
def vendor_return_queue_confirm(unit_id):

    from models import ApplianceUnit

    from services.appliance_issue_service import (
        ApplianceIssueAccessDenied,
        ApplianceIssueError,
        ApplianceIssueService,
    )

    unit = db.session.get(
        ApplianceUnit,
        unit_id,
    )

    if unit is None:

        flash(
            "Appliance not found.",
            "danger",
        )

        return redirect(
            url_for(
                "appliance.vendor_returns_queue"
            )
        )

    if not AccessControlService.can(
        current_user,
        "appliance.vendor_return",
        warehouse_id=unit.warehouse_id,
    ):

        flash(
            "You do not have permission to confirm "
            "Vendor Returns.",
            "danger",
        )

        return redirect(
            url_for(
                "appliance.vendor_returns_queue"
            )
        )

    if (
        unit.status
        or ""
    ).strip().lower() != "vendor_return_pending":

        flash(
            f"{unit.inventory_number} is no longer "
            "Vendor Return Pending.",
            "warning",
        )

        return redirect(
            url_for(
                "appliance.vendor_returns_queue"
            )
        )

    rma_reference = (
        request.form.get(
            "rma_reference"
        )
        or ""
    ).strip()

    return_notes = (
        request.form.get(
            "return_notes"
        )
        or ""
    ).strip()

    receiving = None

    if (
        unit.receiving_line
        and unit.receiving_line.receiving
    ):
        receiving = (
            unit.receiving_line.receiving
        )

    vendor_name = (
        receiving.supplier_name
        if receiving
        else ""
    ) or ""

    # Human-readable audit notes.
    note_parts = []

    if vendor_name:
        note_parts.append(
            f"Vendor: {vendor_name}"
        )

    if rma_reference:
        note_parts.append(
            f"RMA / Ref: {rma_reference}"
        )

    if return_notes:
        note_parts.append(
            return_notes
        )

    movement_notes = "\n".join(
        note_parts
    ) or None

    try:

        ApplianceIssueService.change_inventory_disposition(
            actor=current_user,
            appliance_unit_id=unit.id,
            action="CONFIRM_VENDOR_RETURN",
            reason_code="RETURNED_TO_VENDOR",
            notes=movement_notes,
            extra_movement_meta={
                "vendor_name":
                    vendor_name,

                "rma_reference":
                    rma_reference,

                "return_notes":
                    return_notes,

                "source_receiving_number":
                    (
                        receiving.receiving_number
                        if receiving
                        else ""
                    )
                    or "",

                "source_invoice_number":
                    (
                        receiving.invoice_number
                        if receiving
                        else ""
                    )
                    or "",
            },
        )

        flash(
            f"{unit.inventory_number} returned to "
            f"{vendor_name or 'vendor'}.",
            "success",
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

    return redirect(
        url_for(
            "appliance.vendor_returns_queue"
        )
    )


# ============================================================
# CANCEL PENDING VENDOR RETURN
# ============================================================

@appliance_bp.post(
    "/vendor-returns/<int:unit_id>/cancel"
)
@login_required
def vendor_return_queue_cancel(unit_id):

    from models import ApplianceUnit

    from services.appliance_issue_service import (
        ApplianceIssueAccessDenied,
        ApplianceIssueError,
        ApplianceIssueService,
    )

    unit = db.session.get(
        ApplianceUnit,
        unit_id,
    )

    if unit is None:

        flash(
            "Appliance not found.",
            "danger",
        )

        return redirect(
            url_for(
                "appliance.vendor_returns_queue"
            )
        )

    if not AccessControlService.can(
        current_user,
        "appliance.vendor_return",
        warehouse_id=unit.warehouse_id,
    ):

        flash(
            "You do not have permission to cancel "
            "Vendor Returns.",
            "danger",
        )

        return redirect(
            url_for(
                "appliance.vendor_returns_queue"
            )
        )

    notes = (
        request.form.get(
            "cancel_notes"
        )
        or ""
    ).strip()

    try:

        ApplianceIssueService.change_inventory_disposition(
            actor=current_user,
            appliance_unit_id=unit.id,
            action="CANCEL_VENDOR_RETURN",
            reason_code="RETURN_CANCELLED",
            notes=notes or None,
        )

        flash(
            f"Vendor Return cancelled for "
            f"{unit.inventory_number}. "
            "Appliance is AVAILABLE again.",
            "success",
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

    return redirect(
        url_for(
            "appliance.vendor_returns_queue"
        )
    )


# ============================================================
# VENDOR RETURN - MANAGER REVIEW / ACTION
# ============================================================

@appliance_bp.get(
    "/vendor-returns/<int:unit_id>"
)
@login_required
def vendor_return_action(unit_id):

    from models import (
        ApplianceMovement,
        ApplianceUnit,
    )

    unit = db.session.get(
        ApplianceUnit,
        unit_id,
    )

    if unit is None:

        flash(
            "Appliance not found.",
            "danger",
        )

        return redirect(
            url_for(
                "appliance.vendor_returns_queue"
            )
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
            url_for(
                "appliance.vendor_returns_queue"
            )
        )

    if (
        unit.status
        or ""
    ).strip().lower() != "vendor_return_pending":

        flash(
            f"{unit.inventory_number} is no longer "
            "Vendor Return Pending.",
            "warning",
        )

        return redirect(
            url_for(
                "appliance.inventory_detail",
                unit_id=unit.id,
            )
        )

    pending_movement = (
        ApplianceMovement.query
        .filter(
            ApplianceMovement.appliance_unit_id
            == unit.id,

            ApplianceMovement.movement_type
            == "VENDOR_RETURN_PENDING",
        )
        .order_by(
            ApplianceMovement.created_at.desc(),
            ApplianceMovement.id.desc(),
        )
        .first()
    )

    receiving = None

    if (
        unit.receiving_line
        and unit.receiving_line.receiving
    ):
        receiving = (
            unit.receiving_line.receiving
        )

    vendor_name = (
        receiving.supplier_name
        if receiving
        else ""
    ) or ""

    purchase_invoice = (
        receiving.invoice_number
        if receiving
        else ""
    ) or ""

    receiving_number = (
        receiving.receiving_number
        if receiving
        else ""
    ) or ""

    can_decide = (
        AccessControlService.can(
            current_user,
            "appliance.vendor_return",
            warehouse_id=unit.warehouse_id,
        )
    )

    return render_template(
        "appliance_vendor_return_action.html",

        unit=unit,

        pending_movement=
            pending_movement,

        vendor_name=
            vendor_name,

        purchase_invoice=
            purchase_invoice,

        receiving_number=
            receiving_number,

        can_decide=
            can_decide,
    )


# ============================================================
# RETURN PENDING APPLIANCE AS LOANER
# ============================================================

@appliance_bp.post(
    "/vendor-returns/<int:unit_id>/return-as-loaner"
)
@login_required
def vendor_return_as_loaner(unit_id):

    from models import ApplianceUnit

    from services.appliance_issue_service import (
        ApplianceIssueAccessDenied,
        ApplianceIssueError,
        ApplianceIssueService,
    )

    unit = db.session.get(
        ApplianceUnit,
        unit_id,
    )

    if unit is None:

        flash(
            "Appliance not found.",
            "danger",
        )

        return redirect(
            url_for(
                "appliance.vendor_returns_queue"
            )
        )

    if not AccessControlService.can(
        current_user,
        "appliance.vendor_return",
        warehouse_id=unit.warehouse_id,
    ):

        flash(
            "You do not have permission to manage "
            "Vendor Returns.",
            "danger",
        )

        return redirect(
            url_for(
                "appliance.vendor_returns_queue"
            )
        )

    notes = (
        request.form.get(
            "notes"
        )
        or ""
    ).strip()

    try:

        ApplianceIssueService.change_inventory_disposition(
            actor=current_user,

            appliance_unit_id=
                unit.id,

            action=
                "VENDOR_RETURN_TO_LOANER",

            reason_code=
                "RETURNED_TO_STOCK_AS_LOANER",

            notes=
                notes or None,
        )

        flash(
            f"{unit.inventory_number} returned to "
            "AVAILABLE LOANER stock.",
            "success",
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

        return redirect(
            url_for(
                "appliance.vendor_return_action",
                unit_id=unit.id,
            )
        )

    return redirect(
        url_for(
            "appliance.vendor_returns_queue"
        )
    )


# ============================================================
# WRITE OFF FROM VENDOR RETURN PENDING
# ============================================================

@appliance_bp.post(
    "/vendor-returns/<int:unit_id>/write-off"
)
@login_required
def vendor_return_write_off(unit_id):

    from models import ApplianceUnit

    from services.appliance_issue_service import (
        ApplianceIssueAccessDenied,
        ApplianceIssueError,
        ApplianceIssueService,
    )

    unit = db.session.get(
        ApplianceUnit,
        unit_id,
    )

    if unit is None:

        flash(
            "Appliance not found.",
            "danger",
        )

        return redirect(
            url_for(
                "appliance.vendor_returns_queue"
            )
        )

    if not AccessControlService.can(
        current_user,
        "appliance.vendor_return",
        warehouse_id=unit.warehouse_id,
    ):

        flash(
            "You do not have permission to manage "
            "Vendor Returns.",
            "danger",
        )

        return redirect(
            url_for(
                "appliance.vendor_returns_queue"
            )
        )

    reason_code = (
        request.form.get(
            "reason_code"
        )
        or "VENDOR_RETURN_REJECTED_WRITE_OFF"
    ).strip().upper()

    notes = (
        request.form.get(
            "notes"
        )
        or ""
    ).strip()

    try:

        ApplianceIssueService.change_inventory_disposition(
            actor=current_user,

            appliance_unit_id=
                unit.id,

            action=
                "VENDOR_RETURN_TO_SCRAP",

            reason_code=
                reason_code,

            notes=
                notes or None,
        )

        flash(
            f"{unit.inventory_number} written off.",
            "success",
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

        return redirect(
            url_for(
                "appliance.vendor_return_action",
                unit_id=unit.id,
            )
        )

    return redirect(
        url_for(
            "appliance.vendor_returns_queue"
        )
    )


