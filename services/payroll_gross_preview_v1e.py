
"""WCCR Payroll V1E: guarded illustrative gross preview.

Uses existing V1A-V1D modules.
Does not authorize payroll, modify database or execute payments.
"""

from decimal import Decimal

from services.payroll_calculation_engine_v1a import (
    PayrollBlocked,
    calculate,
)
from services.payroll_data_reader_v1c import (
    load_payroll_sources,
)
from services.payroll_safety_v1d import (
    validate_period,
    validate_overtime_distribution,
)


def preview_from_prepared(
    prepared,
    *,
    verified_empty_dates=(),
):
    """Generate an illustrative estimate, never approved payroll."""

    if not isinstance(prepared, dict):
        raise PayrollBlocked("Invalid prepared payroll data")

    coverage = validate_period(
        prepared,
        verified_empty_dates=verified_empty_dates,
    )

    # Missing daily rows do not prevent an illustrative estimate.
    # They remain unverified; this never authorizes live payroll.
    # The calculation engine receives only validated attendance.

    if not prepared.get("rates"):
        raise PayrollBlocked("Missing pay rate history")

    if not prepared.get("statuses"):
        raise PayrollBlocked("Missing payroll status history")

    result = calculate(
        period_start=prepared["period_start"],
        entries=prepared["entries"],
        rates=prepared["rates"],
        statuses=prepared["statuses"],
        workweek_start=0,
        full_week_context=True,
    )

    validate_overtime_distribution(
        total_hours=coverage["total_hours"],
        regular_hours=result["regular_hours"],
        overtime_hours=result["overtime_hours"],
        doubletime_hours=result["doubletime_hours"],
    )

    total_hours = Decimal("0")
    total_gross = Decimal("0")

    for row in result["details"]:
        total_hours += (
            Decimal(row["regular_hours"])
            + Decimal(row["overtime_hours"])
            + Decimal(row["doubletime_hours"])
        )
        total_gross += Decimal(row["gross_unrounded"])

    if total_hours != Decimal(coverage["total_hours"]):
        raise PayrollBlocked(
            "Daily hour distribution mismatch"
        )

    if abs(
        total_gross - Decimal(result["gross_pay"])
    ) > Decimal("0.005"):
        raise PayrollBlocked(
            "Gross payroll reconciliation failed"
        )

    return {
        **result,
        "status": "ILLUSTRATIVE_GROSS_ESTIMATE",
        "employee_id": prepared["employee_id"],
        "coverage": coverage,
        "source_completeness_verified": False,
        "california_compliance_verified": False,
        "payroll_approved": False,
        "ready_for_payment": False,
        "payment_enabled": False,
        "limitations": result["limitations"] + [
            "Attendance source is preliminary",
            "Empty days are not independently verified",
            "Missing attendance dates are assumed zero only "
            "for this illustrative estimate",
            "No payroll approval",
            "No payment authorization",
        ],
    }


def preview_from_database(
    db,
    *,
    employee_id,
    period_start,
    paid_break_minutes,
    verified_empty_dates=(),
    now=None,
):
    """Read actual WCCR sources for an illustrative estimate.

    Caller must enforce access rights before invoking.
    """

    prepared = load_payroll_sources(
        db,
        employee_id=employee_id,
        period_start=period_start,
        paid_break_minutes=paid_break_minutes,
        now=now,
    )

    return preview_from_prepared(
        prepared,
        verified_empty_dates=verified_empty_dates,
    )


def approve_payroll(*args, **kwargs):
    raise PayrollBlocked(
        "V1E cannot approve payroll"
    )


def execute_payment(*args, **kwargs):
    raise PayrollBlocked(
        "V1E cannot execute payments"
    )
