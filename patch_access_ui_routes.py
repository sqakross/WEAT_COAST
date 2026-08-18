from pathlib import Path

ROOT = Path(".")
THIS_FILE = Path(__file__).resolve()

route_marker = '@inventory_bp.route("/users", methods=["GET", "POST"])'

matches = []

for path in ROOT.rglob("*.py"):
    resolved = path.resolve()

    if resolved == THIS_FILE:
        continue

    if any(
        part in {
            ".venv",
            "venv",
            "migrations",
            "__pycache__",
        }
        for part in path.parts
    ):
        continue

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        continue

    if route_marker in text:
        matches.append(path)

if len(matches) != 1:
    print("ERROR: expected exactly one real routes file.")
    print("Found:", [str(p) for p in matches])
    raise SystemExit(1)

path = matches[0]
text = path.read_text(encoding="utf-8")

backup = path.with_suffix(path.suffix + ".before_access_ui")
backup.write_text(text, encoding="utf-8")

# ============================================================
# 1. Add MANAGER to /users create role whitelist
# ============================================================

old = """        allowed_roles = {
            ROLE_TECHNICIAN,
            ROLE_USER,
            ROLE_VIEWER,
            ROLE_ACCOUNTING,
            ROLE_ADMIN,
            ROLE_SUPERADMIN,
        }
"""

new = """        allowed_roles = {
            ROLE_TECHNICIAN,
            ROLE_USER,
            ROLE_VIEWER,
            ROLE_ACCOUNTING,
            "manager",
            ROLE_ADMIN,
            ROLE_SUPERADMIN,
        }
"""

if old not in text:
    raise SystemExit("ERROR: users() allowed_roles block not found")

text = text.replace(old, new, 1)


# ============================================================
# 2. Add MANAGER to /users role selector
# ============================================================

old = """    role_options = [
        (ROLE_TECHNICIAN, "Technician"),
        (ROLE_USER, "User"),
        (ROLE_VIEWER, "Viewer"),
        (ROLE_ACCOUNTING, "Accounting"),
        (ROLE_ADMIN, "Admin"),
        (ROLE_SUPERADMIN, "Superadmin"),
    ]
"""

new = """    role_options = [
        (ROLE_TECHNICIAN, "Technician"),
        (ROLE_USER, "User"),
        (ROLE_VIEWER, "Viewer"),
        (ROLE_ACCOUNTING, "Accounting"),
        ("manager", "Manager"),
        (ROLE_ADMIN, "Admin"),
        (ROLE_SUPERADMIN, "Superadmin"),
    ]
"""

if old not in text:
    raise SystemExit("ERROR: users() role_options block not found")

text = text.replace(old, new, 1)


# ============================================================
# 3. Add MANAGER to superadmin edit role selector
# ============================================================

old = """        role_options = [
            (ROLE_TECHNICIAN, "Technician"),
            (ROLE_USER, "User"),
            (ROLE_VIEWER, "Viewer"),
            (ROLE_ACCOUNTING, "Accounting"),
            (ROLE_ADMIN, "Admin"),
            (ROLE_SUPERADMIN, "Superadmin"),
        ]
"""

new = """        role_options = [
            (ROLE_TECHNICIAN, "Technician"),
            (ROLE_USER, "User"),
            (ROLE_VIEWER, "Viewer"),
            (ROLE_ACCOUNTING, "Accounting"),
            ("manager", "Manager"),
            (ROLE_ADMIN, "Admin"),
            (ROLE_SUPERADMIN, "Superadmin"),
        ]
"""

if old not in text:
    raise SystemExit("ERROR: edit_user() role_options block not found")

text = text.replace(old, new, 1)


# ============================================================
# 4. Add Access / Permissions route before add_user()
# ============================================================

anchor = """@inventory_bp.route('/users/add', methods=['GET', 'POST'])
@login_required
def add_user():
"""

route = r'''@inventory_bp.route(
    "/users/<int:user_id>/access",
    methods=["GET", "POST"],
)
@login_required
def user_access(user_id):
    """
    Appliance / warehouse access management.

    This permission layer currently applies only to the new
    Appliance Inventory module. Existing ERP role checks remain unchanged.
    """
    from models import (
        Permission,
        UserPermission,
        UserPermissionAudit,
        UserWarehouseAccess,
        Warehouse,
    )
    from services.access_control_service import AccessControlService
    from services.access_management_service import AccessManagementService

    if (current_user.role or "").strip().lower() != ROLE_SUPERADMIN:
        flash("Access denied.", "danger")
        return redirect(url_for("inventory.users"))

    user = User.query.get_or_404(user_id)

    is_target_superadmin = (
        (user.role or "").strip().lower() == ROLE_SUPERADMIN
    )

    warehouses = (
        Warehouse.query
        .filter(Warehouse.is_active.is_(True))
        .order_by(Warehouse.name.asc())
        .all()
    )

    permissions = (
        Permission.query
        .filter(Permission.is_active.is_(True))
        .order_by(
            Permission.group_name.asc(),
            Permission.sort_order.asc(),
            Permission.name.asc(),
        )
        .all()
    )

    if request.method == "POST":
        if is_target_superadmin:
            flash(
                "Superadmin has global access and cannot be restricted here.",
                "warning",
            )
            return redirect(
                url_for(
                    "inventory.user_access",
                    user_id=user.id,
                )
            )

        requested_warehouse_ids = set()

        for raw in request.form.getlist("warehouse_ids"):
            try:
                requested_warehouse_ids.add(int(raw))
            except (TypeError, ValueError):
                continue

        valid_warehouse_ids = {
            int(w.id)
            for w in warehouses
        }

        requested_warehouse_ids &= valid_warehouse_ids

        default_warehouse_id = None

        raw_default = (
            request.form.get("default_warehouse_id")
            or ""
        ).strip()

        if raw_default:
            try:
                candidate = int(raw_default)
            except (TypeError, ValueError):
                candidate = None

            if candidate in requested_warehouse_ids:
                default_warehouse_id = candidate

        current_accesses = {
            int(row.warehouse_id): row
            for row in UserWarehouseAccess.query
            .filter_by(user_id=user.id)
            .all()
        }

        for warehouse in warehouses:
            warehouse_id = int(warehouse.id)

            if warehouse_id in requested_warehouse_ids:
                AccessManagementService.grant_warehouse_access(
                    user=user,
                    warehouse=warehouse,
                    actor=current_user,
                    is_default=(
                        warehouse_id == default_warehouse_id
                    ),
                )
            else:
                existing = current_accesses.get(warehouse_id)

                if existing is not None and existing.is_active:
                    AccessManagementService.revoke_warehouse_access(
                        user=user,
                        warehouse=warehouse,
                        actor=current_user,
                    )

        if requested_warehouse_ids and default_warehouse_id is None:
            first_id = sorted(requested_warehouse_ids)[0]
            first_warehouse = Warehouse.query.get(first_id)

            if first_warehouse is not None:
                AccessManagementService.grant_warehouse_access(
                    user=user,
                    warehouse=first_warehouse,
                    actor=current_user,
                    is_default=True,
                )

        requested_codes = {
            (code or "").strip()
            for code in request.form.getlist("permission_codes")
            if (code or "").strip()
        }

        valid_codes = {
            p.code
            for p in permissions
        }

        requested_codes &= valid_codes

        current_codes = {
            code
            for (code,) in (
                UserPermission.query
                .join(
                    Permission,
                    UserPermission.permission_id == Permission.id,
                )
                .with_entities(Permission.code)
                .filter(UserPermission.user_id == user.id)
                .all()
            )
        }

        for code in sorted(requested_codes - current_codes):
            AccessManagementService.grant_permission(
                user=user,
                permission_code=code,
                actor=current_user,
            )

        for code in sorted(current_codes - requested_codes):
            AccessManagementService.revoke_permission(
                user=user,
                permission_code=code,
                actor=current_user,
            )

        flash(
            f"Access updated for {user.username}.",
            "success",
        )

        return redirect(
            url_for(
                "inventory.user_access",
                user_id=user.id,
            )
        )

    selected_warehouse_ids = {
        int(row.warehouse_id)
        for row in UserWarehouseAccess.query
        .filter(
            UserWarehouseAccess.user_id == user.id,
            UserWarehouseAccess.is_active.is_(True),
        )
        .all()
    }

    default_access = (
        UserWarehouseAccess.query
        .filter(
            UserWarehouseAccess.user_id == user.id,
            UserWarehouseAccess.is_active.is_(True),
            UserWarehouseAccess.is_default.is_(True),
        )
        .first()
    )

    default_warehouse_id = (
        int(default_access.warehouse_id)
        if default_access is not None
        else None
    )

    selected_permission_codes = (
        AccessControlService.permission_codes(user)
        if not is_target_superadmin
        else {
            p.code
            for p in permissions
        }
    )

    permission_groups = {}

    for permission in permissions:
        permission_groups.setdefault(
            permission.group_name or "General",
            [],
        ).append(permission)

    recent_audit = (
        UserPermissionAudit.query
        .filter(
            UserPermissionAudit.target_user_id == user.id
        )
        .order_by(
            UserPermissionAudit.created_at.desc(),
            UserPermissionAudit.id.desc(),
        )
        .limit(30)
        .all()
    )

    return render_template(
        "user_access.html",
        user=user,
        warehouses=warehouses,
        permissions=permissions,
        permission_groups=permission_groups,
        selected_warehouse_ids=selected_warehouse_ids,
        default_warehouse_id=default_warehouse_id,
        selected_permission_codes=selected_permission_codes,
        recent_audit=recent_audit,
        is_target_superadmin=is_target_superadmin,
    )


'''

if anchor not in text:
    raise SystemExit("ERROR: add_user() anchor not found")

text = text.replace(anchor, route + anchor, 1)

path.write_text(text, encoding="utf-8")

print("OK")
print("Routes file:", path)
print("Backup:", backup)
print("Added Manager role option")
print("Added user_access route")
