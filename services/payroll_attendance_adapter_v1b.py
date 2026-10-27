
"""WCCR Payroll Attendance Adapter V1B.

Pure read-only normalization and validation layer.
No database access, no ORM, no Flask routes.
Never authorizes payroll payment or approval.
"""

from datetime import date, datetime, timedelta
from decimal import Decimal

from services.payroll_calculation_engine_v1a import (
    PayrollBlocked,
    calculate,
    number,
    work_date,
)


def field(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def normalize_history(rows, *, kind):
    if not isinstance(rows, (list, tuple)):
        raise PayrollBlocked("Invalid history collection")

    output = []
    seen = set()

    for record in rows:
        effective_date = work_date(field(record, "effective_date"))
        cancelled_at = field(record, "cancelled_at")

        if cancelled_at is not None:
            continue

        if effective_date in seen:
            raise PayrollBlocked(
                "Conflicting effective dates in " + kind
            )
        seen.add(effective_date)

        if kind == "rate":
            amount = number(field(record, "amount"))
            if amount <= 0:
                raise PayrollBlocked("Invalid pay rate")

            output.append({
                "effective_date": effective_date.isoformat(),
                "pay_type": field(record, "pay_type"),
                "classification": field(record, "classification"),
                "amount": str(amount),
            })

        elif kind == "status":
            enabled = field(record, "enabled")
            if type(enabled) is not bool:
                raise PayrollBlocked("Invalid Payroll ON/OFF")

            output.append({
                "effective_date": effective_date.isoformat(),
                "enabled": enabled,
            })

        else:
            raise PayrollBlocked("Unknown history type")

    output.sort(key=lambda x: x["effective_date"])
    return output


def normalize_attendance(
    *,
    dashboard,
    employee_id,
    period_start,
):
    """Validate 14 calendar days of prepared Attendance data.

    Dashboard rows are NOT treated as payroll-approved.
    This function returns normalized hours for inspection only.
    """
    if type(employee_id) is not int or employee_id <= 0:
        raise PayrollBlocked("Invalid employee ID")

    start = work_date(period_start)
    if start.weekday() != 0:
        raise PayrollBlocked("Period must begin Monday")

    if not isinstance(dashboard, dict):
        raise PayrollBlocked("Invalid Attendance dashboard")

    if dashboard.get("preliminary") is not True:
        raise PayrollBlocked("Unexpected Attendance source")

    end = start + timedelta(days=13)

    if dashboard.get("first_day") != start.isoformat():
        raise PayrollBlocked("Attendance start mismatch")

    if dashboard.get("today") != end.isoformat():
        raise PayrollBlocked("Attendance end mismatch")

    rows = dashboard.get("daily_timesheets")
    if not isinstance(rows, list):
        raise PayrollBlocked("Missing daily timesheets")

    dates = set()
    entries = []

    for row in rows:
        if not isinstance(row, dict):
            raise PayrollBlocked("Invalid daily timesheet")

        if row.get("user_id") != employee_id:
            continue

        d = work_date(row.get("date"))
        if d < start or d > end:
            raise PayrollBlocked("Attendance date outside period")

        if d in dates:
            raise PayrollBlocked("Duplicate employee/day")
        dates.add(d)

        if (
            row.get("open") is not False
            or row.get("needs_review") is not False
            or row.get("overlap") is not False
            or row.get("unverified") is not False
        ):
            raise PayrollBlocked(
                "Unresolved Attendance: " + d.isoformat()
            )

        parts = row.get("parts")
        if not isinstance(parts, list) or not parts:
            raise PayrollBlocked("Missing Attendance source parts")

        active = [
            p for p in parts
            if field(p, "status") != "voided"
        ]

        if not active:
            if any(
                field(p, "status") != "voided"
                or field(p, "worked") != 0
                for p in parts
            ):
                raise PayrollBlocked(
                    "Invalid voided-only attendance"
                )
            if row.get("worked") != 0:
                raise PayrollBlocked(
                    "Voided-only day must have zero hours"
                )
            entries.append({
                "date": d.isoformat(),
                "hours": "0",
            })
            continue

        for part in active:
            if field(part, "shift_id") is None:
                raise PayrollBlocked("Missing shift ID")
            if field(part, "open") is not False:
                raise PayrollBlocked("Open source shift")
            if field(part, "requires_review") is not False:
                raise PayrollBlocked("Unapproved source shift")
            if field(part, "unverified") is not False:
                raise PayrollBlocked("Unverified source shift")

        minutes = row.get("worked")
        if type(minutes) is not int or minutes < 0:
            raise PayrollBlocked("Invalid worked minutes")

        # Payroll must not trust aggregate hours without
        # reconciling non-voided source shifts.
        active_minutes = 0
        for part in active:
            part_minutes = field(part, "worked")
            if type(part_minutes) is not int or part_minutes < 0:
                raise PayrollBlocked(
                    "Invalid active shift minutes"
                )
            active_minutes += part_minutes

        for part in parts:
            if field(part, "status") == "voided":
                if field(part, "worked") != 0:
                    raise PayrollBlocked(
                        "Voided shift contains payable minutes"
                    )

        if minutes != active_minutes:
            raise PayrollBlocked(
                "Attendance total differs from active shifts: "
                + d.isoformat()
            )

        hours = Decimal(minutes) / Decimal(3600)
        if hours > Decimal(24):
            raise PayrollBlocked("More than 24 hours in a day")

        entries.append({
            "date": d.isoformat(),
            "hours": str(hours),
        })

    entries.sort(key=lambda x: x["date"])
    return entries


def prepare_preview(
    *,
    dashboard,
    employee_id,
    period_start,
    pay_rates,
    payroll_statuses,
):
    """Prepare validated inputs; do not execute payroll."""
    entries = normalize_attendance(
        dashboard=dashboard,
        employee_id=employee_id,
        period_start=period_start,
    )

    rates = normalize_history(pay_rates, kind="rate")
    statuses = normalize_history(
        payroll_statuses, kind="status"
    )

    return {
        "status": "PREPARED_NOT_APPROVED",
        "employee_id": employee_id,
        "period_start": work_date(period_start).isoformat(),
        "entries": entries,
        "rates": rates,
        "statuses": statuses,
        "payroll_approved": False,
        "ready_for_payment": False,
        "source_completeness_verified": False,
    }


def calculate_preview(*args, **kwargs):
    """Fail closed until source completeness and approval
    validation are implemented in a later integration.
    """
    raise PayrollBlocked(
        "V1B does not authorize payroll calculation from "
        "preliminary Attendance data"
    )
