from services.erp_access_service import ErpAccessService

def can_manage_attendance_users(actor):
    if actor is None:
        return False

    role = (getattr(actor, "role", "") or "").strip().lower()

    if role == "superadmin":
        return True

    if role != "admin":
        return False

    return ErpAccessService.is_allowed(
        actor,
        "erp.attendance.manage_users",
        default_allowed=False,
    )
