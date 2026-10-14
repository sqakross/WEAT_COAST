from datetime import datetime, timezone

from flask import current_app
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from sqlalchemy import select

from extensions import db
from models import User
from services.employee_time_tracking_models import (
    EmployeeShift,
    EmployeeTimeEvent,
    EmployeeTimeSettingsAudit,
)


class EmployeeTimeSettingsError(Exception):
    pass


def utc_now():
    return datetime.now(timezone.utc)


def make_csrf_token(superadmin_id, target_user_id):
    serializer = URLSafeTimedSerializer(
        current_app.secret_key,
        salt="wccr-employee-time-settings-v1",
    )
    return serializer.dumps({
        "admin": int(superadmin_id),
        "target": int(target_user_id),
    })


def verify_csrf_token(token, superadmin_id, target_user_id):
    serializer = URLSafeTimedSerializer(
        current_app.secret_key,
        salt="wccr-employee-time-settings-v1",
    )
    try:
        data = serializer.loads(token, max_age=1800)
    except (BadSignature, SignatureExpired):
        return False

    return (
        data.get("admin") == int(superadmin_id)
        and data.get("target") == int(target_user_id)
    )


def set_tracking_enabled(actor, target_user_id, enabled):
    if actor is None or actor.role != "superadmin":
        raise EmployeeTimeSettingsError("Superadmin required")

    target = db.session.get(User, int(target_user_id))

    if target is None:
        raise EmployeeTimeSettingsError("User not found")

    if target.role == "technician":
        raise EmployeeTimeSettingsError(
            "Technician tracking is prohibited"
        )

    enabled = bool(enabled)
    old_value = bool(target.employee_time_tracking_enabled)

    if old_value == enabled:
        return False

    now = utc_now()

    try:
        if not enabled:
            shifts = db.session.execute(
                select(EmployeeShift).where(
                    EmployeeShift.user_id == target.id,
                    EmployeeShift.ended_at.is_(None),
                )
            ).scalars().all()

            for shift in shifts:
                shift.ended_at = now
                shift.status = "pending_review"
                shift.requires_review = True

                db.session.add(EmployeeTimeEvent(
                    shift_id=shift.id,
                    user_id=target.id,
                    event_type="TRACKING_DISABLED",
                    occurred_at=now,
                    note="Closed by Superadmin tracking disable",
                ))

        target.employee_time_tracking_enabled = enabled

        db.session.add(EmployeeTimeSettingsAudit(
            user_id=target.id,
            changed_by_user_id=actor.id,
            old_enabled=old_value,
            new_enabled=enabled,
            changed_at=now,
        ))

        db.session.commit()
        return True

    except Exception:
        db.session.rollback()
        raise
