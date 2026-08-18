from pathlib import Path

path = Path("inventory/routes.py")
text = path.read_text(encoding="utf-8")

start_marker = "def user_access(user_id):"
start = text.find(start_marker)

if start == -1:
    raise SystemExit("ERROR: def user_access(user_id) not found")

next_route = text.find(
    "\n@inventory_bp.route",
    start + len(start_marker),
)

if next_route == -1:
    raise SystemExit("ERROR: next inventory route not found")

new_function = r'''def user_access(user_id):
    """
    Appliance / warehouse access management.

    This permission layer currently applies only to the new
    Appliance Inventory module. Existing ERP role checks remain unchanged.
    """
    import time

    from flask import current_app

    from models import (
        Permission,
        UserPermission,
        UserPermissionAudit,
        UserWarehouseAccess,
        Warehouse,
    )
    from services.access_control_service import AccessControlService
    from services.access_management_service import AccessManagementService

    perf_start = time.perf_counter()
    perf_last = perf_start
    perf = {}

    def mark(name):
        nonlocal perf_last
        now = time.perf_counter()
        perf[name] = now - perf_last
        perf_last = now

    if (current_user.role or "").strip().lower() != ROLE_SUPERADMIN:
        flash("Access denied.", "danger")
        return redirect(url_for("inventory.users"))

    mark("auth")

    user = User.query.get_or_404(user_id)

    mark("load_user")

    is_target_superadmin = (
        (user.role or "").strip().lower() == ROLE_SUPERADMIN
    )

    warehouses = (
        Warehouse.query
        .filter(Warehouse.is_active.is_(True))
        .order_by(Warehouse.name.asc())
        .all()
    )

    mark("load_warehouses")

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

    mark("load_permissions")

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
            first_warehouse = db.session.get(
                Warehouse,
                first_id,
            )

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

    mark("selected_warehouses")

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

    mark("default_warehouse")

    if is_target_superadmin:
        selected_permission_codes = {
            p.code
            for p in permissions
        }
    else:
        selected_permission_codes = (
            AccessControlService.permission_codes(user)
        )

    mark("permission_codes")

    permission_groups = {}

    for permission in permissions:
        permission_groups.setdefault(
            permission.group_name or "General",
            [],
        ).append(permission)

    mark("group_permissions")

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

    mark("load_audit")

    render_started = time.perf_counter()

    response = render_template(
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

    render_time = time.perf_counter() - render_started
    total_time = time.perf_counter() - perf_start

    current_app.logger.warning(
        "USER_ACCESS PERF user=%s total=%.3fs | "
        "auth=%.3fs | "
        "load_user=%.3fs | "
        "load_warehouses=%.3fs | "
        "load_permissions=%.3fs | "
        "selected_warehouses=%.3fs | "
        "default_warehouse=%.3fs | "
        "permission_codes=%.3fs | "
        "group_permissions=%.3fs | "
        "load_audit=%.3fs | "
        "render=%.3fs",
        user_id,
        total_time,
        perf.get("auth", 0.0),
        perf.get("load_user", 0.0),
        perf.get("load_warehouses", 0.0),
        perf.get("load_permissions", 0.0),
        perf.get("selected_warehouses", 0.0),
        perf.get("default_warehouse", 0.0),
        perf.get("permission_codes", 0.0),
        perf.get("group_permissions", 0.0),
        perf.get("load_audit", 0.0),
        render_time,
    )

    return response
'''

text = (
    text[:start]
    + new_function
    + text[next_route:]
)

path.write_text(text, encoding="utf-8")

print("OK: user_access PERF instrumentation installed")
print("File:", path)
