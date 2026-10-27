
"""WCCR Payroll Calculation Engine V1A.

Pure Python, read-only calculation kernel.
No database, Flask, ORM, approvals or payments.
Not a complete California payroll compliance engine.
"""

from datetime import date, datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP

ZERO = Decimal("0")
EIGHT = Decimal("8")
TWELVE = Decimal("12")
FORTY = Decimal("40")
CENT = Decimal("0.01")


class PayrollBlocked(ValueError):
    pass


def number(value):
    if isinstance(value, bool):
        raise PayrollBlocked("Boolean numeric input")
    try:
        result = Decimal(str(value))
    except Exception as exc:
        raise PayrollBlocked("Invalid numeric input") from exc
    if not result.is_finite():
        raise PayrollBlocked("Non-finite numeric input")
    return result


def work_date(value):
    if isinstance(value, datetime):
        raise PayrollBlocked("Use local work date, not datetime")
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            pass
    raise PayrollBlocked("Invalid work date")


def cents(value):
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def effective(rows, target):
    selected = []
    for row in rows:
        if not isinstance(row, dict):
            raise PayrollBlocked("Invalid history record")
        if row.get("cancelled", False):
            continue
        d = work_date(row["effective_date"])
        if d <= target:
            selected.append((d, row))

    if not selected:
        return None

    selected.sort(key=lambda item: item[0])
    if len(selected) > 1 and selected[-1][0] == selected[-2][0]:
        raise PayrollBlocked("Duplicate effective-date records")

    return selected[-1][1]


def calculate(
    *,
    period_start,
    entries,
    rates,
    statuses,
    workweek_start=0,
    full_week_context=False,
):
    start = work_date(period_start)

    if type(workweek_start) is not int or workweek_start != 0:
        raise PayrollBlocked("WCCR workweek must start Monday")

    if start.weekday() != 0:
        raise PayrollBlocked("Biweekly period must start Monday")

    if full_week_context is not True:
        raise PayrollBlocked("Complete workweek not certified")

    if not isinstance(entries, list):
        raise PayrollBlocked("Invalid entries")
    if not isinstance(rates, list) or not isinstance(statuses, list):
        raise PayrollBlocked("Invalid pay history")

    end = start + timedelta(days=13)
    daily = {}

    for entry in entries:
        if not isinstance(entry, dict):
            raise PayrollBlocked("Invalid attendance entry")
        d = work_date(entry["date"])
        hours = number(entry["hours"])

        if d < start or d > end:
            raise PayrollBlocked("Hours outside period")
        if hours < ZERO or hours > Decimal("24"):
            raise PayrollBlocked("Invalid daily hours")
        daily[d] = daily.get(d, ZERO) + hours
        if daily[d] > Decimal("24"):
            raise PayrollBlocked("Daily hours exceed 24")

    # Fail closed on rate changes within a workweek with
    # payable hours: blended regular-rate rules are not yet
    # implemented. Same rate across a full workweek is safe
    # for this limited hourly calculation model.
    for week in range(2):
        days = [start + timedelta(days=week * 7 + i)
                for i in range(7)]
        active_rates = set()
        for d in days:
            if daily.get(d, ZERO) <= ZERO:
                continue
            rate = effective(rates, d)
            if rate is None:
                raise PayrollBlocked("Missing effective rate")
            active_rates.add((
                rate.get("pay_type"),
                rate.get("classification"),
                str(number(rate.get("amount"))),
            ))
        if len(active_rates) > 1:
            raise PayrollBlocked(
                "Multiple pay rates in one workweek require "
                "regular-rate calculation"
            )

    details = []
    gross = ZERO
    regular_total = ZERO
    ot_total = ZERO
    dt_total = ZERO

    for week in range(2):
        weekly_regular = ZERO
        consecutive = 0

        for i in range(7):
            d = start + timedelta(days=week * 7 + i)
            hours = daily.get(d, ZERO)

            status = effective(statuses, d)
            enabled = (
                status.get("enabled") is True
                if status is not None else False
            )

            if hours > ZERO and not enabled:
                raise PayrollBlocked(
                    "Payable work exists while Payroll OFF: "
                    + d.isoformat()
                )

            if hours <= ZERO:
                consecutive = 0
                continue

            rate = effective(rates, d)
            if rate is None:
                raise PayrollBlocked("Missing pay rate")

            if (
                rate.get("pay_type") != "hourly"
                or rate.get("classification") != "non_exempt"
            ):
                raise PayrollBlocked(
                    "Only hourly non-exempt supported"
                )

            hourly = number(rate["amount"])
            if hourly <= ZERO:
                raise PayrollBlocked("Invalid hourly rate")

            consecutive += 1

            if consecutive == 7:
                # CA seventh consecutive day in workweek:
                # first 8 hours 1.5x, above 8 hours 2x.
                regular = ZERO
                overtime = min(hours, EIGHT)
                doubletime = max(ZERO, hours - EIGHT)
            else:
                daily_double = max(ZERO, hours - TWELVE)
                daily_ot = min(
                    max(ZERO, hours - EIGHT),
                    Decimal("4"),
                )
                regular_candidate = min(hours, EIGHT)

                capacity = max(ZERO, FORTY - weekly_regular)
                regular = min(regular_candidate, capacity)
                weekly_ot = regular_candidate - regular

                overtime = daily_ot + weekly_ot
                doubletime = daily_double

            # Only non-daily-overtime hours count toward
            # the 40-hour weekly straight-time threshold.
            if consecutive != 7:
                weekly_regular += min(hours, EIGHT)

            pay = (
                regular * hourly
                + overtime * hourly * Decimal("1.5")
                + doubletime * hourly * Decimal("2")
            )

            gross += pay
            regular_total += regular
            ot_total += overtime
            dt_total += doubletime

            details.append({
                "date": d.isoformat(),
                "rate": str(hourly),
                "regular_hours": str(regular),
                "overtime_hours": str(overtime),
                "doubletime_hours": str(doubletime),
                "gross_unrounded": str(pay),
            })

    return {
        "status": "PREVIEW_ONLY",
        "period_start": start.isoformat(),
        "period_end": end.isoformat(),
        "regular_hours": str(regular_total),
        "overtime_hours": str(ot_total),
        "doubletime_hours": str(dt_total),
        "gross_pay": str(cents(gross)),
        "details": details,
        "limitations": [
            "Source attendance verification required",
            "No payroll approval or payment",
            "No bonuses, commissions or adjustments",
            "No blended regular-rate calculations",
            "No salary or exempt pay",
            "No meal/rest premiums",
            "No taxes or deductions",
        ],
    }
