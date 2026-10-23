"""WCCR V7.3C - save attendance correction revisions.

Original EmployeeTimeEvent rows are never changed.
This service does not approve attendance or payroll.
"""

import hashlib
import json
from datetime import datetime, timezone

from sqlalchemy import select, func
from sqlalchemy.exc import IntegrityError

from extensions import db
from services.attendance_review_v73 import (
    AttendanceReviewRevision,
    AttendanceReviewPeriod,
    AttendanceReviewError,
    parse_pacific,
    validate_revision,
    aware_utc,
)
from services.employee_time_tracking_models import (
    EmployeeShift,
    EmployeeTimeEvent,
    EmployeeTimeAudit,
)


def source_fingerprint(shift, events):
    """Stable fingerprint of source shift and original events."""
    payload = {
        "shift": shift.id,
        "user": shift.user_id,
        "start": aware_utc(shift.started_at).isoformat(),
        "end": aware_utc(shift.ended_at).isoformat(),
        "events": [
            [
                event.id,
                event.event_type,
                aware_utc(event.occurred_at).isoformat(),
                event.note or "",
            ]
            for event in sorted(events, key=lambda x: x.id)
        ],
    }
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def latest_revision_number(shift_id):
    value = db.session.execute(
        select(func.max(AttendanceReviewRevision.revision))
        .where(AttendanceReviewRevision.shift_id == shift_id)
    ).scalar_one()
    return int(value or 0)


def save_revision(
    *,
    shift_id,
    actor_id,
    expected_revision,
    expected_source,
    started_at,
    ended_at,
    periods,
    reason,
    commit=True,
):
    """Persist a pending revision and audit in one transaction."""
    if type(expected_revision) is not int or expected_revision < 0:
        raise AttendanceReviewError("Invalid revision number.")

    if (
        not isinstance(expected_source, str)
        or len(expected_source) != 64
    ):
        raise AttendanceReviewError("Invalid source version.")

    if not isinstance(periods, list):
        raise AttendanceReviewError("Invalid periods payload.")

    normalized = []

    for item in periods:
        if not isinstance(item, dict):
            raise AttendanceReviewError("Invalid period.")

        if set(item) != {"kind", "start", "end", "source_event_ids"}:
            raise AttendanceReviewError(
                "Unexpected period fields."
            )

        ids = item["source_event_ids"]

        if not isinstance(ids, list) or len(ids) > 2:
            raise AttendanceReviewError(
                "Invalid source event references."
            )

        if any(type(value) is not int or value <= 0 for value in ids):
            raise AttendanceReviewError(
                "Invalid event reference."
            )

        normalized.append({
            "kind": item["kind"],
            "start": parse_pacific(item["start"]),
            "end": parse_pacific(item["end"]),
            "source_event_ids": ids,
        })

    result = validate_revision(
        parse_pacific(started_at),
        parse_pacific(ended_at),
        normalized,
        reason,
    )

    try:
        shift = db.session.get(EmployeeShift, shift_id)

        if shift is None or shift.ended_at is None:
            raise AttendanceReviewError(
                "Closed shift not found."
            )

        if shift.status not in ("closed", "pending_review"):
            raise AttendanceReviewError(
                "Shift status does not permit editing."
            )

        events = db.session.execute(
            select(EmployeeTimeEvent)
            .where(EmployeeTimeEvent.shift_id == shift_id)
            .order_by(EmployeeTimeEvent.id)
        ).scalars().all()

        if not events:
            raise AttendanceReviewError(
                "Shift has no original events."
            )

        if source_fingerprint(shift, events) != expected_source:
            raise AttendanceReviewError(
                "Original shift changed. Reload the page."
            )

        current = latest_revision_number(shift_id)

        if current != expected_revision:
            raise AttendanceReviewError(
                "Another correction was saved. Reload the page."
            )

        valid_event_ids = {event.id for event in events}
        referenced_ids = []

        for item in normalized:
            for event_id in item["source_event_ids"]:
                if event_id not in valid_event_ids:
                    raise AttendanceReviewError(
                        "Period references another shift."
                    )
                referenced_ids.append(event_id)

        if len(referenced_ids) != len(set(referenced_ids)):
            raise AttendanceReviewError(
                "Source event used in multiple periods."
            )

        # Existing events are retained. A revision is a separate
        # correction record, not an automatic payroll adjustment.
        new_revision = AttendanceReviewRevision(
            shift_id=shift_id,
            revision=current + 1,
            status="pending",
            started_at_utc=result["start"].replace(tzinfo=None),
            ended_at_utc=result["end"].replace(tzinfo=None),
            created_by_user_id=actor_id,
            reason=result["reason"],
        )

        db.session.add(new_revision)
        db.session.flush()

        for item in normalized:
            db.session.add(AttendanceReviewPeriod(
                revision_id=new_revision.id,
                kind=item["kind"],
                started_at_utc=item["start"].replace(tzinfo=None),
                ended_at_utc=item["end"].replace(tzinfo=None),
                source_event_ids=json.dumps(
                    item["source_event_ids"]
                ),
            ))

        previous = db.session.execute(
            select(AttendanceReviewRevision)
            .where(
                AttendanceReviewRevision.shift_id == shift_id,
                AttendanceReviewRevision.revision == current,
            )
        ).scalar_one_or_none()

        old_value = (
            "revision=" + str(previous.revision)
            if previous is not None
            else "original events"
        )

        db.session.add(EmployeeTimeAudit(
            shift_id=shift_id,
            changed_by_user_id=actor_id,
            field_name="attendance_review_revision",
            old_value=old_value,
            new_value=json.dumps({
                "revision": current + 1,
                "status": "pending",
                "clock_in": result["start"].isoformat(),
                "clock_out": result["end"].isoformat(),
                "periods": [
                    {
                        "kind": item["kind"],
                        "start": item["start"].isoformat(),
                        "end": item["end"].isoformat(),
                        "source_event_ids": item["source_event_ids"],
                    }
                    for item in normalized
                ],
            }, separators=(",", ":")),
            reason=result["reason"],
        ))

        # A correction is not approval.
        # Do not modify source shift, event rows or payroll.
        if commit:
            db.session.commit()
        else:
            db.session.flush()
        return new_revision.id, current + 1

    except IntegrityError as exc:
        db.session.rollback()
        raise AttendanceReviewError(
            "Concurrent revision detected. Reload the page."
        ) from exc
    except Exception:
        db.session.rollback()
        raise
