from pathlib import Path

path = Path("inventory/routes.py")
text = path.read_text(encoding="utf-8")

old = '''    recent_audit = (
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
'''

new = '''    recent_audit_rows = (
        UserPermissionAudit.query
        .filter(
            UserPermissionAudit.target_user_id == user.id
        )
        .order_by(
            UserPermissionAudit.created_at.desc(),
            UserPermissionAudit.id.desc(),
        )
        .limit(100)
        .all()
    )

    # ---------------------------------------------------------
    # Group noisy permission audit rows for UI.
    #
    # Detailed rows remain untouched in DB.
    # UI groups changes made by the same actor within the same
    # minute into one readable access-update event.
    # ---------------------------------------------------------
    audit_groups_map = {}

    for row in recent_audit_rows:
        created_at = row.created_at

        minute_key = (
            created_at.replace(
                second=0,
                microsecond=0,
            )
            if created_at is not None
            else None
        )

        key = (
            minute_key,
            row.actor_username or "",
        )

        group = audit_groups_map.get(key)

        if group is None:
            group = {
                "created_at": created_at,
                "created_at_local": row.created_at_local,
                "actor_username": row.actor_username,
                "permission_granted": [],
                "permission_revoked": [],
                "warehouse_granted": [],
                "warehouse_revoked": [],
                "warehouse_updated": [],
                "other": [],
            }

            audit_groups_map[key] = group

        action = (row.action or "").strip().upper()

        if action == "PERMISSION_GRANTED":
            if row.permission_code:
                group["permission_granted"].append(
                    row.permission_code
                )

        elif action == "PERMISSION_REVOKED":
            if row.permission_code:
                group["permission_revoked"].append(
                    row.permission_code
                )

        elif action == "WAREHOUSE_ACCESS_GRANTED":
            warehouse_code = (
                row.warehouse.code
                if row.warehouse is not None
                else str(row.warehouse_id or "")
            )

            if warehouse_code:
                group["warehouse_granted"].append(
                    warehouse_code
                )

        elif action == "WAREHOUSE_ACCESS_REVOKED":
            warehouse_code = (
                row.warehouse.code
                if row.warehouse is not None
                else str(row.warehouse_id or "")
            )

            if warehouse_code:
                group["warehouse_revoked"].append(
                    warehouse_code
                )

        elif action == "WAREHOUSE_ACCESS_UPDATED":
            warehouse_code = (
                row.warehouse.code
                if row.warehouse is not None
                else str(row.warehouse_id or "")
            )

            if warehouse_code:
                group["warehouse_updated"].append(
                    warehouse_code
                )

        else:
            group["other"].append(
                {
                    "action": row.action,
                    "permission_code": row.permission_code,
                    "warehouse": (
                        row.warehouse.code
                        if row.warehouse is not None
                        else None
                    ),
                }
            )

    recent_audit_groups = list(
        audit_groups_map.values()
    )[:30]

    return render_template(
'''

if old not in text:
    raise SystemExit("ERROR: recent_audit block not found")

text = text.replace(old, new, 1)

old = '''        recent_audit=recent_audit,
        is_target_superadmin=is_target_superadmin,
'''

new = '''        recent_audit_groups=recent_audit_groups,
        is_target_superadmin=is_target_superadmin,
'''

if old not in text:
    raise SystemExit("ERROR: render recent_audit argument not found")

text = text.replace(old, new, 1)

path.write_text(text, encoding="utf-8")

print("OK: audit grouping backend installed")
