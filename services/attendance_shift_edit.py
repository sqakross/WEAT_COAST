"""WCCR Edit Shift V4.0 - restricted completed-shift editor."""
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import select, update
from extensions import db
from services.employee_time_tracking_models import (
    EmployeeShift, EmployeeTimeEvent, EmployeeTimeAudit,
)

LA = ZoneInfo("America/Los_Angeles")
ALLOWED_EVENTS = {"LOGIN", "END_SHIFT", "IDLE"}


class ShiftEditError(ValueError):
    pass


def aware(value):
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def local_string(value):
    # WCCR_OPEN_SHIFT_NONE_FIX_V54
    if value is None:
        return ""
    return aware(value).astimezone(LA).strftime("%Y-%m-%dT%H:%M")


def parse_local(value):
    try:
        naive = datetime.strptime(value, "%Y-%m-%dT%H:%M")
    except (TypeError, ValueError):
        raise ShiftEditError("Invalid date or time.")

    # Reject nonexistent and ambiguous DST local times.
    candidates = []
    for fold in (0, 1):
        local = naive.replace(tzinfo=LA, fold=fold)
        utc = local.astimezone(timezone.utc)
        if utc.astimezone(LA).replace(tzinfo=None) == naive:
            candidates.append(utc)
    unique = set(candidates)
    if len(unique) != 1:
        raise ShiftEditError(
            "Ambiguous or nonexistent California time."
        )
    return candidates[0]


def validate_shift(shift, events):
    if shift is None:
        raise ShiftEditError("Shift not found.")
    if shift.ended_at is None or shift.status not in (
        "closed", "pending_review"
    ):
        raise ShiftEditError("Only completed shifts can be edited.")
    if not events:
        raise ShiftEditError("Shift has no events.")
    types = [event.event_type for event in events]
    if (
        any(t not in ALLOWED_EVENTS for t in types)
        or types.count("LOGIN") != 1
        or types.count("END_SHIFT") != 1
    ):
        raise ShiftEditError(
            "This shift needs the advanced interval editor."
        )
    if aware(shift.ended_at) <= aware(shift.started_at):
        raise ShiftEditError("Existing shift interval is invalid.")


def get_shift(shift_id):
    shift = db.session.get(EmployeeShift, shift_id)
    events = db.session.execute(
        select(EmployeeTimeEvent)
        .where(EmployeeTimeEvent.shift_id == shift_id)
        .order_by(EmployeeTimeEvent.id)
    ).scalars().all()
    validate_shift(shift, events)
    return shift, events


def save_shift(shift_id, actor_id, start_text, end_text,
               reason, expected_start, expected_end):
    reason = (reason or "").strip()
    if len(reason) < 10 or len(reason) > 1000:
        raise ShiftEditError(
            "Reason must contain 10 to 1000 characters."
        )

    start = parse_local(start_text)
    end = parse_local(end_text)
    if end <= start or end - start > timedelta(hours=24):
        raise ShiftEditError(
            "Shift must be positive and no longer than 24 hours."
        )
    if end > datetime.now(timezone.utc):
        raise ShiftEditError("Shift end cannot be in the future.")

    try:
        shift, events = get_shift(shift_id)
        old_start = aware(shift.started_at)
        old_end = aware(shift.ended_at)

        if (
            old_start.isoformat() != expected_start
            or old_end.isoformat() != expected_end
        ):
            raise ShiftEditError(
                "Shift changed. Reload before editing."
            )

        # WCCR_EDIT_SHIFT_V52_PRECISION
        # Original events remain unchanged. datetime-local inputs
        # contain minutes, while stored events include seconds.
        # Treat the selected Clock Out minute as inclusive for
        # validation, without changing the submitted end timestamp.
        event_validation_end = end + timedelta(minutes=1)

        for event in events:
            at = aware(event.occurred_at)
            if at < start or at >= event_validation_end:
                raise ShiftEditError(
                    "An original event is outside the proposed "
                    "interval. Advanced editing is required."
                )

        if start == old_start and end == old_end:
            raise ShiftEditError("No changes were made.")

        # Conditional UPDATE prevents a stale form from overwriting
        # another edit. Status and original times must still match.
        result = db.session.execute(
            update(EmployeeShift)
            .where(
                EmployeeShift.id == shift_id,
                EmployeeShift.started_at == shift.started_at,
                EmployeeShift.ended_at == shift.ended_at,
                EmployeeShift.status == shift.status,
            )
            .values(
                started_at=start.replace(tzinfo=None),
                ended_at=end.replace(tzinfo=None),
                status="pending_review",
                requires_review=True,
            )
            .execution_options(synchronize_session=False)
        )
        if result.rowcount != 1:
            raise ShiftEditError(
                "Shift changed concurrently. Reload."
            )

        for field, old, new in (
            ("started_at", old_start, start),
            ("ended_at", old_end, end),
        ):
            if old != new:
                db.session.add(EmployeeTimeAudit(
                    shift_id=shift_id,
                    changed_by_user_id=actor_id,
                    field_name=field,
                    old_value=old.isoformat(),
                    new_value=new.isoformat(),
                    reason=reason,
                ))

        db.session.commit()
    except Exception:
        db.session.rollback()
        raise
