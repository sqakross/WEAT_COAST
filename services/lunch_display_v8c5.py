
"""WCCR V8C.5A: unified calculated Lunch display.

Read-only calculation. Does not approve or modify payroll.
"""

def calculate_daily_display(
    *,
    shift_seconds,
    recorded_lunch_seconds,
    required_lunch_seconds,
    auto_lunch_enabled,
    other_unpaid_seconds=0,
):
    values = (
        shift_seconds,
        recorded_lunch_seconds,
        required_lunch_seconds,
        other_unpaid_seconds,
    )

    if any(type(x) is not int or x < 0 for x in values):
        raise ValueError("Invalid time values")

    if type(auto_lunch_enabled) is not bool:
        raise ValueError("Invalid Auto Lunch setting")

    if recorded_lunch_seconds > shift_seconds:
        raise ValueError("Lunch exceeds shift")

    if (
        recorded_lunch_seconds + other_unpaid_seconds
        > shift_seconds
    ):
        raise ValueError("Unpaid time exceeds shift")

    if auto_lunch_enabled:
        lunch = max(
            recorded_lunch_seconds,
            required_lunch_seconds,
        )
    else:
        lunch = recorded_lunch_seconds

    if lunch + other_unpaid_seconds > shift_seconds:
        raise ValueError("Calculated lunch exceeds shift")

    worked = (
        shift_seconds
        - lunch
        - other_unpaid_seconds
    )

    return {
        "shift": shift_seconds,
        "lunch": lunch,
        "worked": worked,
        "calculation_only": True,
        "payroll_approved": False,
    }
