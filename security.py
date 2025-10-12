# security.py
from flask_login import current_user
from models import ROLE_SUPERADMIN, ROLE_ADMIN, ROLE_TECHNICIAN

def role() -> str:
    return (getattr(current_user, "role", "") or "").strip().lower()

def is_superadmin() -> bool:
    return role() == ROLE_SUPERADMIN

def is_admin() -> bool:
    return role() in (ROLE_ADMIN, ROLE_SUPERADMIN)

def is_technician() -> bool:
    return role() == ROLE_TECHNICIAN
