
"""WCCR V7.3 - immutable-source attendance correction foundation.

This module defines the correction records and validates proposed
time intervals. Database migration, application routes and approval
integration are installed separately.
"""

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from extensions import db

PACIFIC = ZoneInfo("America/Los_Angeles")


class AttendanceReviewError(ValueError):
    pass


class AttendanceReviewRevision(db.Model):
    __tablename__ = "employee_time_review_revisions"

    id = db.Column(db.Integer, primary_key=True)
    shift_id = db.Column(
        db.Integer, nullable=False, index=True
    )
    revision = db.Column(db.Integer, nullable=False)
    status = db.Column(
        db.String(30), nullable=False, default="pending"
    )

    started_at_utc = db.Column(
        db.DateTime(timezone=True), nullable=False
    )
    ended_at_utc = db.Column(
        db.DateTime(timezone=True), nullable=False
    )

    created_by_user_id = db.Column(
        db.Integer, nullable=False
    )
    created_at_utc = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc)
    )

    approved_by_user_id = db.Column(
        db.Integer, nullable=True
    )
    approved_at_utc = db.Column(
        db.DateTime(timezone=True), nullable=True
    )

    reason = db.Column(db.Text, nullable=False)

    __table_args__ = (
        db.UniqueConstraint(
            "shift_id",
            "revision",
            name="uq_employee_time_review_revision"
        ),
    )


class AttendanceReviewPeriod(db.Model):
    __tablename__ = "employee_time_review_periods"

    id = db.Column(db.Integer, primary_key=True)
    revision_id = db.Column(
        db.Integer,
        db.ForeignKey("employee_time_review_revisions.id"),
        nullable=False,
        index=True
    )

    kind = db.Column(db.String(10), nullable=False)
    started_at_utc = db.Column(
        db.DateTime(timezone=True), nullable=False
    )
    ended_at_utc = db.Column(
        db.DateTime(timezone=True), nullable=False
    )

    source_event_ids = db.Column(
        db.Text, nullable=True
    )


def aware_utc(value):
    if value is None:
        raise AttendanceReviewError("Missing timestamp.")
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def parse_pacific(value):
    """Reject nonexistent and ambiguous Pacific local times."""
    if not isinstance(value, str):
        raise AttendanceReviewError("Invalid date/time.")

    try:
        naive = datetime.strptime(
            value, "%Y-%m-%dT%H:%M"
        )
    except ValueError as exc:
        raise AttendanceReviewError(
            "Use YYYY-MM-DDTHH:MM."
        ) from exc

    candidates = set()

    for fold in (0, 1):
        local = naive.replace(
            tzinfo=PACIFIC, fold=fold
        )
        instant = local.astimezone(timezone.utc)

        if (
            instant.astimezone(PACIFIC)
            .replace(tzinfo=None) == naive
        ):
            candidates.add(instant)

    if len(candidates) != 1:
        raise AttendanceReviewError(
            "Ambiguous or nonexistent Pacific time."
        )

    return candidates.pop()


def validate_revision(
    started_at,
    ended_at,
    periods,
    reason,
    *,
    now=None
):
    """Validate correction data without modifying original events."""
    start = aware_utc(started_at)
    end = aware_utc(ended_at)

    now = aware_utc(
        now or datetime.now(timezone.utc)
    )

    if not start < end:
        raise AttendanceReviewError(
            "Clock Out must be after Clock In."
        )

    if end - start > timedelta(hours=24):
        raise AttendanceReviewError(
            "Shift exceeds 24 hours. Manual review required."
        )

    if end > now:
        raise AttendanceReviewError(
            "Clock Out cannot be in the future."
        )

    reason = (reason or "").strip()

    if not 4 <= len(reason) <= 1000:
        raise AttendanceReviewError(
            "Reason must contain 4-1000 characters."
        )

    if not isinstance(periods, list):
        raise AttendanceReviewError(
            "Invalid Break/Lunch data."
        )

    if len(periods) > 50:
        raise AttendanceReviewError(
            "Too many Break/Lunch intervals."
        )

    validated = []

    for item in periods:
        if not isinstance(item, dict):
            raise AttendanceReviewError(
                "Invalid interval."
            )

        kind = item.get("kind")

        if kind not in ("Break", "Lunch"):
            raise AttendanceReviewError(
                "Invalid interval type."
            )

        left = aware_utc(item.get("start"))
        right = aware_utc(item.get("end"))

        if not start <= left < right <= end:
            raise AttendanceReviewError(
                "Break/Lunch must be within the shift."
            )

        validated.append({
            "kind": kind,
            "start": left,
            "end": right,
        })

    validated.sort(
        key=lambda item: (
            item["start"], item["end"]
        )
    )

    for previous, current in zip(
        validated, validated[1:]
    ):
        if current["start"] < previous["end"]:
            raise AttendanceReviewError(
                "Break/Lunch intervals overlap."
            )

    return {
        "start": start,
        "end": end,
        "periods": validated,
        "reason": reason,
    }
