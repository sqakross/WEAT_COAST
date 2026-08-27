from pathlib import Path
from datetime import datetime

path = Path("appliance/routes.py")

if not path.exists():
    raise SystemExit("ERROR: appliance/routes.py not found")

text = path.read_text(encoding="utf-8")

backup = Path(
    f"appliance/routes.py.before_issue_ui_"
    f"{datetime.now():%Y%m%d_%H%M%S}.bak"
)
backup.write_text(text, encoding="utf-8")

if "def issue_new(" in text:
    raise SystemExit(
        "STOP: Appliance Issue UI routes already exist."
    )

block = r'''

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

            work_order_id = int(
                request.form.get("work_order_id")
                or 0
            )

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
                work_order_id=work_order_id,
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

    from sqlalchemy import or_
    from models import (
        ApplianceCategory,
        ApplianceUnit,
    )

    q = (
        request.args.get("q")
        or ""
    ).strip()

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
                "items": [],
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
            ApplianceCategory.sort_order.asc(),
            ApplianceCategory.name.asc(),
            ApplianceUnit.inventory_number.asc(),
        )
        .limit(30)
        .all()
    )

    return jsonify(
        {
            "ok": True,
            "items": [
                {
                    "id": unit.id,
                    "inventory_number":
                        unit.inventory_number,

                    "appliance_type":
                        (
                            unit.category.name.upper()
                            if unit.category
                            else ""
                        ),

                    "brand":
                        (unit.brand or "").upper(),

                    "model":
                        (
                            unit.model_number
                            or ""
                        ).upper(),

                    "serial":
                        (
                            unit.serial_number
                            or ""
                        ).upper(),

                    "size":
                        (
                            (
                                f"{unit.size_value:g} "
                                f"{unit.size_unit or ''}"
                            ).strip()
                            if unit.size_value
                            is not None
                            else ""
                        ),

                    "condition":
                        (
                            unit.condition
                            or ""
                        ).replace(
                            "_",
                            " ",
                        ).upper(),
                }
                for unit in units
            ],
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

'''

text = text.rstrip() + "\n" + block + "\n"

path.write_text(
    text,
    encoding="utf-8",
)

print("=" * 70)
print("APPLIANCE ISSUE UI ROUTES")
print("=" * 70)
print("OK: routes added")
print("  /appliances/issues/new")
print("  /appliances/issues/search-appliances")
print("  /appliances/issues/search-work-orders")
print("  /appliances/issues/<id>")
print()
print("Backup:", backup)
print("=" * 70)
