
"""WCCR V3.0.1 - Employee Pay Profiles.

Models only. No schema creation or database writes on import.
"""
from datetime import datetime, timezone
from extensions import db


def utc_now():
    return datetime.now(timezone.utc)


class EmployeePayProfile(db.Model):
    __tablename__ = "employee_pay_profiles"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(
        db.Integer,
        db.ForeignKey("user.id"),
        nullable=False,
        unique=True,
        index=True,
    )
    work_state = db.Column(
        db.String(2), nullable=False, default="CA"
    )
    pay_type = db.Column(db.String(10), nullable=False)
    exemption_status = db.Column(
        db.String(15), nullable=False
    )
    created_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )
    updated_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )


class EmployeePayRateHistory(db.Model):
    __tablename__ = "employee_pay_rate_history"

    id = db.Column(db.Integer, primary_key=True)
    profile_id = db.Column(
        db.Integer,
        db.ForeignKey("employee_pay_profiles.id"),
        nullable=False,
        index=True,
    )
    effective_from = db.Column(db.Date, nullable=False)
    effective_to = db.Column(db.Date, nullable=True)
    amount = db.Column(db.Numeric(14, 2), nullable=False)
    pay_type = db.Column(db.String(10), nullable=False)
    exemption_status = db.Column(db.String(15), nullable=False)
    work_state = db.Column(db.String(2), nullable=False)
    created_by_id = db.Column(
        db.Integer,
        db.ForeignKey("user.id"),
        nullable=False,
    )
    created_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )
    __table_args__ = (
        db.CheckConstraint(
            "amount >= 0",
            name="ck_employee_pay_rate_nonnegative",
        ),
        db.CheckConstraint(
            "effective_to IS NULL OR "
            "effective_to >= effective_from",
            name="ck_employee_pay_rate_dates",
        ),
    )


class EmployeePayAudit(db.Model):
    __tablename__ = "employee_pay_audit"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(
        db.Integer,
        db.ForeignKey("user.id"),
        nullable=False,
        index=True,
    )
    actor_id = db.Column(
        db.Integer,
        db.ForeignKey("user.id"),
        nullable=False,
    )
    action = db.Column(db.String(50), nullable=False)
    old_values = db.Column(db.Text, nullable=True)
    new_values = db.Column(db.Text, nullable=False)
    created_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )
