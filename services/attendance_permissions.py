from services.erp_access_service import ErpAccessService

def can_manage_attendance_users(actor):
    # WCCR_FLEX_ACCESS_V14A
    # Explicit individual ALLOW only, regardless of base role.
    # Superadmin has no operational bypass.
    # WCCR_SUPERADMIN_FULL_V14C_AST
    if actor is not None and (getattr(actor, 'role', '') or '').strip().lower() == 'superadmin':
        return True
    if actor is None:
        return False

    if getattr(actor, "is_active", True) is False:
        return False

    from services.erp_access_service import ErpAccessService

    return (
        ErpAccessService.get_override(
            actor,
            "erp.attendance.manage_users",
        ) == ErpAccessService.EFFECT_ALLOW
    )
