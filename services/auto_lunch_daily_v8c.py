
"""WCCR Auto Lunch V8C: daily reconciliation, no payroll mutation."""

from datetime import date


def reconcile_daily_lunch(parts, *, work_date, today):
    """
    Compute one daily meal estimate.

    Recorded meal time remains the only meal time eligible for
    unpaid-hour accounting. Proposed minutes are informational.
    """

    from services.employee_time_calculations import (
        daily_auto_lunch_preview_for_parts,
    )

    day = date.fromisoformat(work_date)
    current = date.fromisoformat(today)

    if day > current:
        raise ValueError("Future work date")

    active = [
        p for p in parts
        if p.get("status") != "voided"
    ]

    if not active:
        return {
            "status": "EMPTY",
            "worked_seconds": 0,
            "lunch_seconds": 0,
            "proposed_lunch_seconds": 0,
            "final_day": day < current,
            "payroll_eligible": False,
        }

    blocked = any(
        p.get("open")
        or p.get("unverified")
        or p.get("requires_review")
        or p.get("worked") is None
        for p in active
    )

    worked = sum(
        max(0, int(p.get("worked") or 0))
        for p in active
    )

    lunch = sum(
        max(0, int(p.get("lunch") or 0))
        for p in active
    )

    final_day = day < current

    if blocked:
        proposed = 0
        status = "REVIEW_REQUIRED"
    else:
        proposed = daily_auto_lunch_preview_for_parts(active)
        status = "FINAL" if final_day else "PRELIMINARY"

    return {
        "status": status,
        "worked_seconds": worked,
        "lunch_seconds": lunch,
        "proposed_lunch_seconds": proposed,
        "final_day": final_day,
        "payroll_eligible": bool(final_day and not blocked),
    }
