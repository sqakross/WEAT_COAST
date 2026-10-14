"""WCCR Midnight Auto-Close Engine V1.

Not automatically scheduled or registered.
All timestamps are UTC; day boundaries use America/Los_Angeles.
"""
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from sqlalchemy import select, update
from extensions import db
from services.employee_time_tracking_models import (
    EmployeeShift, EmployeeTimeEvent, EmployeeTimeAudit
)

LA = ZoneInfo("America/Los_Angeles")
EVENT_TYPE = "AUTO_CLOSE_MIDNIGHT"


def utc_aware(value):
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def next_midnight_utc(start):
    local_day = utc_aware(start).astimezone(LA).date()
    following = local_day + timedelta(days=1)
    return datetime(
        following.year, following.month, following.day,
        tzinfo=LA
    ).astimezone(timezone.utc)


def close_expired_shift(shift_id, now=None):
    """Close one overdue open shift, atomically and with audit.

    Must run inside Flask app context.
    Caller must not have unrelated pending DB changes.
    Returns True only if this call closed the shift.
    """
    now = utc_aware(now or datetime.now(timezone.utc))

    try:
        shift = db.session.get(EmployeeShift, shift_id)
        if shift is None or shift.ended_at is not None:
            return False

        boundary = next_midnight_utc(shift.started_at)
        if now < boundary:
            return False

        if utc_aware(shift.started_at) >= boundary:
            raise RuntimeError("Invalid midnight boundary")

        result = db.session.execute(
            update(EmployeeShift)
            .where(
                EmployeeShift.id == shift_id,
                EmployeeShift.ended_at.is_(None)
            )
            .values(
                ended_at=boundary,
                status="pending_review",
                requires_review=True
            )
            .execution_options(synchronize_session=False)
        )

        if result.rowcount != 1:
            db.session.rollback()
            return False

        db.session.add(EmployeeTimeEvent(
            shift_id=shift.id,
            user_id=shift.user_id,
            event_type=EVENT_TYPE,
            occurred_at=boundary,
            note="System midnight closure; admin review required"
        ))

        db.session.add(EmployeeTimeAudit(
            shift_id=shift.id,
            changed_by_user_id=0,
            field_name="ended_at",
            old_value=None,
            new_value=boundary.isoformat(),
            reason="SYSTEM: Midnight auto-close; pending admin review"
        ))

        db.session.commit()
        return True

    except Exception:
        db.session.rollback()
        raise


def close_expired_shifts(now=None):
    """Reconcile all overdue shifts; returns number closed.

    Call only from an explicitly configured scheduler/maintenance
    process, not from an ordinary GET request.
    """
    now = utc_aware(now or datetime.now(timezone.utc))
    ids = db.session.execute(
        select(EmployeeShift.id)
        .where(EmployeeShift.ended_at.is_(None))
        .order_by(EmployeeShift.id)
    ).scalars().all()

    closed = 0
    for shift_id in ids:
        if close_expired_shift(shift_id, now=now):
            closed += 1
    return closed
