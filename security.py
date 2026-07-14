# security.py
from flask_login import current_user

from models import (
    ROLE_SUPERADMIN,
    ROLE_ADMIN,
    ROLE_ACCOUNTING,
    ROLE_USER,
    ROLE_VIEWER,
    ROLE_TECHNICIAN,
)
def role() -> str:
    return (getattr(current_user, "role", "") or "").strip().lower()

def is_superadmin() -> bool:
    return role() == ROLE_SUPERADMIN

def is_admin() -> bool:
    return role() in (ROLE_ADMIN, ROLE_SUPERADMIN)

def is_technician() -> bool:
    return role() == ROLE_TECHNICIAN

def current_role():
    return (getattr(current_user, "role", "") or "").strip().lower()


def is_accounting():
    return current_role() == ROLE_ACCOUNTING


def can_modify_operational_data():
    return current_role() in (
        ROLE_ADMIN,
        ROLE_SUPERADMIN,
    )


def can_view_operational_data():
    return current_role() in (
        ROLE_VIEWER,
        ROLE_ACCOUNTING,
        ROLE_ADMIN,
        ROLE_SUPERADMIN,
    )


def can_use_reports():
    return current_role() in (
        ROLE_ACCOUNTING,
        ROLE_ADMIN,
        ROLE_SUPERADMIN,
    )
