"""CSRF protection for Attendance Review permission updates."""

import hmac
import secrets

from flask import current_app, session
from itsdangerous import (
    URLSafeTimedSerializer,
    BadSignature,
    SignatureExpired,
)

SALT = "wccr-attendance-review-permissions-v1"
NONCE_KEY = "wccr_attendance_review_csrf_nonce"


def _nonce():
    value = session.get(NONCE_KEY)
    if not isinstance(value, str) or len(value) < 32:
        value = secrets.token_urlsafe(32)
        session[NONCE_KEY] = value
    return value


def make_csrf_token(admin_id, target_id):
    serializer = URLSafeTimedSerializer(
        current_app.secret_key, salt=SALT
    )
    return serializer.dumps({
        "admin": int(admin_id),
        "target": int(target_id),
        "nonce": _nonce(),
    })


def verify_csrf_token(token, admin_id, target_id):
    nonce = session.get(NONCE_KEY)
    if not isinstance(nonce, str) or len(nonce) < 32:
        return False
    serializer = URLSafeTimedSerializer(
        current_app.secret_key, salt=SALT
    )
    try:
        data = serializer.loads(token, max_age=1800)
    except (BadSignature, SignatureExpired, TypeError, ValueError):
        return False
    return (
        isinstance(data, dict)
        and data.get("admin") == int(admin_id)
        and data.get("target") == int(target_id)
        and isinstance(data.get("nonce"), str)
        and hmac.compare_digest(data["nonce"], nonce)
    )


def rotate_csrf_nonce():
    session[NONCE_KEY] = secrets.token_urlsafe(32)
