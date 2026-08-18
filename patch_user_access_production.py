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
    from models import (
        Permission,
        UserPermission,
        UserPermissionAudit,
        UserWarehouseAccess,
        Warehouse,
    )
    from services.access_control_service import AccessControlService
    from services.access_management_service import AccessManagementService

    # ---------------------------------------------------------
    # Only superadmin may configure access at this stage.
    # ---------------------------------------------------------
    if (current_user.role or "").strip().lower() != ROLE_SUPERADMIN:
        flash("Access denied.", "danger")
        return redirect(url_for("inventory.users"))

    user = User.query.get_or_404(user_id)

    is_target_superadmin = (
        (user.role or "").strip().lower() == ROLE_SUPERADMIN
    )

    # ---------------------------------------------------------
    # Active warehouses
    # ---------------------------------------------------------
    warehouses = (
        Warehouse.query
        .filter(Warehouse.is_active.is_(True))
        .order_by(Warehouse.name.asc())
        .all()
    )

    # ---------------------------------------------------------
    # Active permissions
    # ---------------------------------------------------------
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

    # =========================================================
    # SAVE ACCESS
    # =========================================================
    if request.method == "POST":

        # Superadmin always has global bypass.
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

        # -----------------------------------------------------
        # Requested warehouses
        # -----------------------------------------------------
        requested_warehouse_ids = set()

        for raw in request.form.getlist("warehouse_ids"):
            try:
                requested_warehouse_ids.add(int(raw))
            except (TypeError, ValueError):
                continue

        valid_warehouse_ids = {
            int(warehouse.id)
            for warehouse in warehouses
        }

        requested_warehouse_ids &= valid_warehouse_ids

        # -----------------------------------------------------
        # Requested default warehouse
        # -----------------------------------------------------
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

        # -----------------------------------------------------
        # Current warehouse access
        # -----------------------------------------------------
        current_accesses = {
            int(row.warehouse_id): row
            for row in (
                UserWarehouseAccess.query
                .filter_by(user_id=user.id)
                .all()
            )
        }

        # -----------------------------------------------------
        # Synchronize warehouse scope
        # -----------------------------------------------------
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

                if (
                    existing is not None
                    and existing.is_active
                ):
                    AccessManagementService.revoke_warehouse_access(
                        user=user,
                        warehouse=warehouse,
                        actor=current_user,
                    )

        # If warehouses are selected but user did not explicitly
        # choose a default, use the first selected warehouse.
        if (
            requested_warehouse_ids
            and default_warehouse_id is None
        ):
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

        # -----------------------------------------------------
        # Requested permissions
        # -----------------------------------------------------
        requested_codes = {
            (code or "").strip()
            for code in request.form.getlist(
                "permission_codes"
            )
            if (code or "").strip()
        }

        valid_codes = {
            permission.code
            for permission in permissions
        }

        requested_codes &= valid_codes

        # -----------------------------------------------------
        # Current permissions
        # -----------------------------------------------------
        current_codes = {
            code
            for (code,) in (
                UserPermission.query
                .join(
                    Permission,
                    UserPermission.permission_id
                    == Permission.id,
                )
                .with_entities(Permission.code)
                .filter(
                    UserPermission.user_id == user.id
                )
                .all()
            )
        }

        # -----------------------------------------------------
        # Grant new permissions
        # -----------------------------------------------------
        for code in sorted(
            requested_codes - current_codes
        ):
            AccessManagementService.grant_permission(
                user=user,
                permission_code=code,
                actor=current_user,
            )

        # -----------------------------------------------------
        # Revoke removed permissions
        # -----------------------------------------------------
        for code in sorted(
            current_codes - requested_codes
        ):
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

    # =========================================================
    # LOAD CURRENT ACCESS FOR UI
    # =========================================================

    active_accesses = (
        UserWarehouseAccess.query
        .filter(
            UserWarehouseAccess.user_id == user.id,
            UserWarehouseAccess.is_active.is_(True),
        )
        .all()
    )

    selected_warehouse_ids = {
        int(row.warehouse_id)
        for row in active_accesses
    }

    default_warehouse_id = next(
        (
            int(row.warehouse_id)
            for row in active_accesses
            if row.is_default
        ),
        None,
    )

    # ---------------------------------------------------------
    # Permission selection
    # ---------------------------------------------------------
    if is_target_superadmin:
        selected_permission_codes = {
            permission.code
            for permission in permissions
        }
    else:
        selected_permission_codes = (
            AccessControlService.permission_codes(user)
        )

    # ---------------------------------------------------------
    # Group permissions for UI
    # ---------------------------------------------------------
    permission_groups = {}

    for permission in permissions:
        permission_groups.setdefault(
            permission.group_name or "General",
            [],
        ).append(permission)

    # ---------------------------------------------------------
    # Recent access audit
    # ---------------------------------------------------------
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

text = (
    text[:start]
    + new_function
    + text[next_route:]
)

path.write_text(text, encoding="utf-8")

print("OK: production user_access installed")
print("Removed temporary PERF instrumentation")
print("Kept optimized access loading")
