
"""WCCR V7.3F: approve a saved attendance revision.

Attendance approval only. No payroll approval or original
event mutation. Permission is checked by the calling route.
"""

import json
from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from extensions import db
from services.attendance_review_v73 import (
    AttendanceReviewRevision,
    AttendanceReviewPeriod,
    AttendanceReviewError,
    validate_revision,
    aware_utc,
)
from services.attendance_review_save_v73 import (
    source_fingerprint,
)
from services.employee_time_tracking_models import (
    EmployeeShift,
    EmployeeTimeEvent,
    EmployeeTimeAudit,
)


def approve_revision(
    *,
    shift_id,
    actor_id,
    revision_id,
    expected_source,
    commit=True,
):
    if type(revision_id) is not int or revision_id <= 0:
        raise AttendanceReviewError("Invalid revision ID.")

    if not isinstance(expected_source, str) or len(expected_source) != 64:
        raise AttendanceReviewError("Invalid source fingerprint.")

    try:
        shift = db.session.get(EmployeeShift, shift_id)

        if shift is None or shift.ended_at is None:
            raise AttendanceReviewError("Closed shift not found.")

        if shift.status != "pending_review":
            raise AttendanceReviewError(
                "Shift is not pending review."
            )

        events = db.session.execute(
            select(EmployeeTimeEvent)
            .where(EmployeeTimeEvent.shift_id == shift_id)
            .order_by(EmployeeTimeEvent.id)
        ).scalars().all()

        if source_fingerprint(shift, events) != expected_source:
            raise AttendanceReviewError(
                "Original shift changed. Reload the page."
            )

        latest = db.session.execute(
            select(AttendanceReviewRevision)
            .where(AttendanceReviewRevision.shift_id == shift_id)
            .order_by(
                AttendanceReviewRevision.revision.desc(),
                AttendanceReviewRevision.id.desc()
            )
            .limit(1)
        ).scalar_one_or_none()

        if latest is None or latest.id != revision_id:
            raise AttendanceReviewError(
                "A newer revision exists. Reload the page."
            )

        if latest.status != "pending":
            raise AttendanceReviewError(
                "Revision is not pending approval."
            )

        saved_periods = db.session.execute(
            select(AttendanceReviewPeriod)
            .where(AttendanceReviewPeriod.revision_id == latest.id)
            .order_by(
                AttendanceReviewPeriod.started_at_utc,
                AttendanceReviewPeriod.id
            )
        ).scalars().all()

        periods = [
            {
                "kind": item.kind,
                "start": aware_utc(item.started_at_utc),
                "end": aware_utc(item.ended_at_utc),
            }
            for item in saved_periods
        ]

        validated = validate_revision(
            latest.started_at_utc,
            latest.ended_at_utc,
            periods,
            latest.reason,
        )

        # Atomic conditional update prevents approving twice.
        now = datetime.now(timezone.utc).replace(tzinfo=None)

        result = db.session.execute(
            update(AttendanceReviewRevision)
            .where(
                AttendanceReviewRevision.id == revision_id,
                AttendanceReviewRevision.shift_id == shift_id,
                AttendanceReviewRevision.status == "pending",
            )
            .values(
                status="approved",
                approved_by_user_id=actor_id,
                approved_at_utc=now,
            )
        )

        if result.rowcount != 1:
            raise AttendanceReviewError(
                "Revision was already changed. Reload."
            )

        db.session.add(EmployeeTimeAudit(
            shift_id=shift_id,
            changed_by_user_id=actor_id,
            field_name="attendance_review_approval",
            old_value="pending",
            new_value=json.dumps({
                "revision_id": revision_id,
                "revision": latest.revision,
                "status": "approved",
                "clock_in": validated["start"].isoformat(),
                "clock_out": validated["end"].isoformat(),
                "period_count": len(periods),
                "approved_by": actor_id,
                "approved_at": now.isoformat(),
            }),
            reason="Attendance revision approval: " + latest.reason,
        ))

        # Preserve original shift/events and all payroll data.
        if commit:
            db.session.commit()
        else:
            db.session.flush()

        return {
            "revision_id": revision_id,
            "revision": latest.revision,
            "status": "approved",
        }

    except IntegrityError as exc:
        db.session.rollback()
        raise AttendanceReviewError(
            "Concurrent approval detected."
        ) from exc
    except Exception:
        db.session.rollback()
        raise
