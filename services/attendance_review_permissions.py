"""
WCCR Employee Time Review - explicit permissions.

These permissions are deliberately independent of the ERP
Superadmin automatic-allow behavior.

No permission is granted by role alone.
"""

from services.erp_permission_registry import ERP_PERMISSION_CODES

REVIEW_PERMISSIONS = {
    "review": "erp.attendance.review_shifts",
    "edit": "erp.attendance.edit_shifts",
    "approve": "erp.attendance.approve_shifts",
    "audit": "erp.attendance.view_audit",
}


def can_review_action(actor, action, *, session=None):
    """
    Explicit ALLOW only.

    No Superadmin bypass, no implicit role permission,
    no default access and no automatic inheritance.
    """
    # WCCR_SUPERADMIN_FULL_V14C_AST
    if actor is not None and (getattr(actor, 'role', '') or '').strip().lower() == 'superadmin':
        return True
    if actor is None:
        return False

    permission_code = REVIEW_PERMISSIONS.get(action)
    if permission_code is None:
        return False

    if permission_code not in ERP_PERMISSION_CODES:
        return False

    user_id = getattr(actor, "id", None)
    if user_id is None:
        return False

    if getattr(actor, "is_active", True) is False:
        return False

    # WCCR_ATTENDANCE_V67A_STRICT
    # Every actor, including Superadmin, requires explicit ALLOW.

    if session is None:
        from extensions import db
        session = db.session

    from models import ErpAccessOverride

    row = (
        session.query(ErpAccessOverride)
        .filter_by(
            user_id=user_id,
            permission_code=permission_code,
        )
        .first()
    )

    return (
        row is not None
        and (getattr(row, "effect", "") or "").strip().upper()
        == "ALLOW"
    )


def require_review_action(actor, action, *, session=None):
    """Raise PermissionError when explicit permission is absent."""
    if not can_review_action(actor, action, session=session):
        raise PermissionError(
            "Employee Time Review permission denied."
        )
    return True
