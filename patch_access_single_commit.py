from pathlib import Path

path = Path("inventory/routes.py")
text = path.read_text(encoding="utf-8")

replacements = [
    (
'''                AccessManagementService.grant_warehouse_access(
                    user=user,
                    warehouse=warehouse,
                    actor=current_user,
                    is_default=(
                        warehouse_id == default_warehouse_id
                    ),
                )''',
'''                AccessManagementService.grant_warehouse_access(
                    user=user,
                    warehouse=warehouse,
                    actor=current_user,
                    is_default=(
                        warehouse_id == default_warehouse_id
                    ),
                    commit=False,
                )'''
    ),
    (
'''                    AccessManagementService.revoke_warehouse_access(
                        user=user,
                        warehouse=warehouse,
                        actor=current_user,
                    )''',
'''                    AccessManagementService.revoke_warehouse_access(
                        user=user,
                        warehouse=warehouse,
                        actor=current_user,
                        commit=False,
                    )'''
    ),
    (
'''                AccessManagementService.grant_warehouse_access(
                    user=user,
                    warehouse=first_warehouse,
                    actor=current_user,
                    is_default=True,
                )''',
'''                AccessManagementService.grant_warehouse_access(
                    user=user,
                    warehouse=first_warehouse,
                    actor=current_user,
                    is_default=True,
                    commit=False,
                )'''
    ),
    (
'''            AccessManagementService.grant_permission(
                user=user,
                permission_code=code,
                actor=current_user,
            )''',
'''            AccessManagementService.grant_permission(
                user=user,
                permission_code=code,
                actor=current_user,
                commit=False,
            )'''
    ),
    (
'''            AccessManagementService.revoke_permission(
                user=user,
                permission_code=code,
                actor=current_user,
            )''',
'''            AccessManagementService.revoke_permission(
                user=user,
                permission_code=code,
                actor=current_user,
                commit=False,
            )'''
    ),
]

for old, new in replacements:
    if old not in text:
        raise SystemExit(
            "ERROR: expected block not found:\n" + old
        )

    text = text.replace(old, new, 1)


old = '''        flash(
            f"Access updated for {user.username}.",
            "success",
        )

        return redirect(
'''

new = '''        # One atomic commit for the complete access update.
        try:
            db.session.commit()
        except Exception as exc:
            db.session.rollback()

            flash(
                f"Failed to update access: {exc}",
                "danger",
            )

            return redirect(
                url_for(
                    "inventory.user_access",
                    user_id=user.id,
                )
            )

        flash(
            f"Access updated for {user.username}.",
            "success",
        )

        return redirect(
'''

if old not in text:
    raise SystemExit(
        "ERROR: final access flash block not found"
    )

text = text.replace(old, new, 1)

path.write_text(text, encoding="utf-8")

print("OK: user_access now uses one transaction")
