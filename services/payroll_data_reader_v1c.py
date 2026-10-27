
"""WCCR Payroll V1C: read-only database source adapter.

Reads real Attendance and Payroll Settings.
Never commits, flushes, approves, calculates wages or pays.
"""

from datetime import date, timedelta

from sqlalchemy import select

from models import EmployeePayRate, EmployeePayrollStatus
from services.attendance_dashboard_service import (
    get_attendance_dashboard_data,
)
from services.payroll_attendance_adapter_v1b import (
    PayrollBlocked,
    prepare_preview,
)


def load_payroll_sources(
    db,
    *,
    employee_id,
    period_start,
    paid_break_minutes,
    now=None,
    dashboard=None,
):
    if type(employee_id) is not int or employee_id <= 0:
        raise PayrollBlocked("Invalid employee ID")

    if type(period_start) is not date:
        raise PayrollBlocked("Period start must be a date")

    if period_start.weekday() != 0:
        raise PayrollBlocked("Period must begin Monday")

    if (
        type(paid_break_minutes) is not int
        or paid_break_minutes < 0
    ):
        raise PayrollBlocked("Invalid paid break allowance")

    period_end = period_start + timedelta(days=13)

    if now is None:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)

    if now.tzinfo is None:
        raise PayrollBlocked("Current time must be timezone-aware")

    from zoneinfo import ZoneInfo
    today_la = now.astimezone(
        ZoneInfo("America/Los_Angeles")
    ).date()

    if period_end > today_la:
        raise PayrollBlocked("Future payroll period is incomplete")

    # Existing WCCR Attendance service is the only source
    # of effective corrected daily timesheets.
    if dashboard is None:
        dashboard = get_attendance_dashboard_data(
            db,
            paid_break_minutes=paid_break_minutes,
            can_review=True,
            now=now,
            start_date=period_start,
            end_date=period_end,
        )

    rates = db.session.execute(
        select(EmployeePayRate)
        .where(EmployeePayRate.user_id == employee_id)
        .where(EmployeePayRate.effective_date <= period_end)
        .order_by(
            EmployeePayRate.effective_date,
            EmployeePayRate.id,
        )
    ).scalars().all()

    statuses = db.session.execute(
        select(EmployeePayrollStatus)
        .where(EmployeePayrollStatus.user_id == employee_id)
        .where(EmployeePayrollStatus.effective_date <= period_end)
        .order_by(
            EmployeePayrollStatus.effective_date,
            EmployeePayrollStatus.id,
        )
    ).scalars().all()

    prepared = prepare_preview(
        dashboard=dashboard,
        employee_id=employee_id,
        period_start=period_start,
        pay_rates=rates,
        payroll_statuses=statuses,
    )

    # An empty attendance result does not prove the employee
    # had zero hours. A separate completeness check is required.
    prepared["source_completeness_verified"] = False
    prepared["payroll_approved"] = False
    prepared["ready_for_payment"] = False
    prepared["status"] = "SOURCE_LOADED_NOT_APPROVED"

    return prepared


def calculate_live_payroll(*args, **kwargs):
    raise PayrollBlocked(
        "Live payroll calculation remains disabled. "
        "Attendance completeness and approval not verified."
    )
