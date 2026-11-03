"""WCCR Auto Lunch policy - pure calculation layer.

This module calculates scheduled meal minutes only.
It does not itself subtract wages, create events, or assert
that a meal was actually taken.

Integration must preserve actual worked time and existing
attendance corrections.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AutoLunchResult:
    enabled: bool
    gross_minutes: int
    first_meal_minutes: int
    second_meal_minutes: int
    total_meal_minutes: int


def calculate_auto_lunch(
    gross_minutes,
    *,
    enabled=False,
    first_meal_waived=False,
    second_meal_waived=False,
):
    """Calculate configured automatic meal allocation.

    WCCR policy:
      - Under 6 hours: no automatic first meal allocation.
      - 6 hours or more: first 30 minutes.
      - Over 10 hours: second 30 minutes, unless waived.

    Waiver inputs must come from verified effective-dated
    employee settings, not from an assumed default.
    """
    if type(gross_minutes) is not int or gross_minutes < 0:
        raise ValueError("gross_minutes must be a nonnegative integer")

    for name, value in (
        ("enabled", enabled),
        ("first_meal_waived", first_meal_waived),
        ("second_meal_waived", second_meal_waived),
    ):
        if type(value) is not bool:
            raise ValueError(name + " must be boolean")

    first = 0
    second = 0

    if enabled:
        if gross_minutes >= 360 and not first_meal_waived:
            first = 30
        if gross_minutes > 600 and not second_meal_waived:
            second = 30

    return AutoLunchResult(
        enabled=enabled,
        gross_minutes=gross_minutes,
        first_meal_minutes=first,
        second_meal_minutes=second,
        total_meal_minutes=first + second,
    )
