"""Employee time tracking database models.

These models are not registered or migrated until the integration patch.
Technician activity tracking is separate and must remain unchanged.
"""

from datetime import datetime, timezone
from extensions import db


def utc_now():
    return datetime.now(timezone.utc)


class EmployeeShift(db.Model):
    __tablename__ = "employee_time_shifts"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False, index=True)

    started_at = db.Column(db.DateTime(timezone=True), nullable=False)
    ended_at = db.Column(db.DateTime(timezone=True), nullable=True)

    status = db.Column(db.String(30), nullable=False, default="open")
    requires_review = db.Column(db.Boolean, nullable=False, default=False)

    created_at = db.Column(
        db.DateTime(timezone=True), nullable=False, default=utc_now
    )


class EmployeeTimeEvent(db.Model):
    __tablename__ = "employee_time_events"

    id = db.Column(db.Integer, primary_key=True)
    shift_id = db.Column(db.Integer, nullable=False, index=True)
    user_id = db.Column(db.Integer, nullable=False, index=True)

    event_type = db.Column(db.String(40), nullable=False)
    occurred_at = db.Column(
        db.DateTime(timezone=True), nullable=False, default=utc_now
    )

    note = db.Column(db.Text, nullable=True)


class EmployeeTimeAudit(db.Model):
    __tablename__ = "employee_time_audit"

    id = db.Column(db.Integer, primary_key=True)
    shift_id = db.Column(db.Integer, nullable=False, index=True)
    changed_by_user_id = db.Column(db.Integer, nullable=False)

    field_name = db.Column(db.String(80), nullable=False)
    old_value = db.Column(db.Text, nullable=True)
    new_value = db.Column(db.Text, nullable=True)
    reason = db.Column(db.Text, nullable=False)

    changed_at = db.Column(
        db.DateTime(timezone=True), nullable=False, default=utc_now
    )


class EmployeeTimeSettingsAudit(db.Model):
    __tablename__ = "employee_time_settings_audit"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False, index=True)
    changed_by_user_id = db.Column(db.Integer, nullable=False)
    old_enabled = db.Column(db.Boolean, nullable=False)
    new_enabled = db.Column(db.Boolean, nullable=False)
    changed_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )
