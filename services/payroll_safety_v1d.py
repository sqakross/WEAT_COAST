
"""WCCR Payroll V1D: isolated safety validation.

Does not approve payroll, query database or authorize payment.
"""

from datetime import date, timedelta
from decimal import Decimal, InvalidOperation


class PayrollSafetyError(ValueError):
    pass


def _decimal(value):
    if isinstance(value, bool):
        raise PayrollSafetyError("Boolean is not a numeric value")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        raise PayrollSafetyError("Invalid numeric value")
    if not result.is_finite() or result < 0:
        raise PayrollSafetyError("Negative/nonfinite value")
    return result


def _date(value):
    if type(value) is date:
        return value
    if type(value) is str:
        try:
            return date.fromisoformat(value)
        except ValueError:
            pass
    raise PayrollSafetyError("Invalid work date")


def validate_period(prepared, *, verified_empty_dates=()):
    """Validate full 14-day coverage.

    Empty dates require independent authoritative confirmation.
    Caller assertions alone do not authorize live payroll.
    """
    if not isinstance(prepared, dict):
        raise PayrollSafetyError("Invalid prepared data")

    if prepared.get("ready_for_payment") is not False:
        raise PayrollSafetyError("Unsafe payment state")

    if prepared.get("payroll_approved") is not False:
        raise PayrollSafetyError("Unexpected payroll approval")

    if prepared.get("source_completeness_verified") is not False:
        raise PayrollSafetyError("Unexpected source verification")

    start = _date(prepared.get("period_start"))
    if start.weekday() != 0:
        raise PayrollSafetyError("Period must start Monday")

    expected = {
        start + timedelta(days=i) for i in range(14)
    }

    entries = prepared.get("entries")
    if not isinstance(entries, list):
        raise PayrollSafetyError("Invalid attendance entries")

    seen = set()
    total = Decimal("0")

    for entry in entries:
        if not isinstance(entry, dict):
            raise PayrollSafetyError("Invalid attendance entry")

        day = _date(entry.get("date"))
        if day not in expected:
            raise PayrollSafetyError("Date outside period")
        if day in seen:
            raise PayrollSafetyError("Duplicate work date")
        seen.add(day)

        hours = _decimal(entry.get("hours"))
        if hours > 24:
            raise PayrollSafetyError("More than 24 hours per day")
        total += hours

    if not isinstance(verified_empty_dates, (tuple, list, set)):
        raise PayrollSafetyError("Invalid empty-date collection")

    empty = [_date(day) for day in verified_empty_dates]
    if len(empty) != len(set(empty)):
        raise PayrollSafetyError("Duplicate empty-date assertion")

    empty = set(empty)
    if empty & seen:
        raise PayrollSafetyError("Worked day marked empty")
    if empty - expected:
        raise PayrollSafetyError("Empty date outside period")

    missing = expected - seen - empty

    return {
        "period_start": start.isoformat(),
        "period_end": (
            start + timedelta(days=13)
        ).isoformat(),
        "days_with_attendance": len(seen),
        "asserted_empty_days": len(empty),
        "unresolved_days": len(missing),
        "total_hours": str(total),
        "complete_coverage_asserted": not missing,
        "authoritatively_verified": False,
        "payroll_approved": False,
        "ready_for_payment": False,
        "missing_dates": [
            d.isoformat() for d in sorted(missing)
        ],
    }


def validate_overtime_distribution(
    *,
    total_hours,
    regular_hours,
    overtime_hours,
    doubletime_hours,
):
    """Verify mutually exclusive hour buckets.

    This is an arithmetic safety check, NOT a determination
    of California legal overtime eligibility.
    """
    total = _decimal(total_hours)
    regular = _decimal(regular_hours)
    overtime = _decimal(overtime_hours)
    doubletime = _decimal(doubletime_hours)

    if regular + overtime + doubletime != total:
        raise PayrollSafetyError(
            "Overtime buckets do not reconcile to total hours"
        )

    return {
        "total_hours": str(total),
        "regular_hours": str(regular),
        "overtime_hours": str(overtime),
        "doubletime_hours": str(doubletime),
        "buckets_reconciled": True,
        "california_compliance_verified": False,
        "ready_for_payment": False,
    }


def authorize_live_payroll(*args, **kwargs):
    raise PayrollSafetyError(
        "V1D cannot authorize live payroll or payment"
    )
