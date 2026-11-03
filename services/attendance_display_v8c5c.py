
"""WCCR FIX4.1 - Attendance presentation only."""

from services.lunch_display_v8c5 import (
    calculate_daily_display,
)


def prepare_attendance_display(source):
    result = dict(source)
    rows = []

    for original in source.get("daily_timesheets", []):
        row = dict(original)

        worked = row.get("worked")
        lunch = row.get("lunch")
        additional = row.get(
            "auto_lunch_proposed_seconds", 0
        )

        safe = (
            row.get("auto_lunch_status") == "FINAL"
            and row.get("open") is False
            and row.get("needs_review") is False
            and row.get("overlap") is False
            and row.get("unverified") is False
            and type(worked) is int
            and type(lunch) is int
            and type(additional) is int
            and worked >= 0
            and lunch >= 0
            and additional > 0
            and additional <= worked
        )

        if safe:
            calculated = calculate_daily_display(
                shift_seconds=worked + lunch,
                recorded_lunch_seconds=lunch,
                required_lunch_seconds=lunch + additional,
                auto_lunch_enabled=True,
            )

            new_worked = calculated["worked"]
            new_lunch = calculated["lunch"]

            if (
                type(new_worked) is int
                and type(new_lunch) is int
                and new_worked >= 0
                and new_lunch >= lunch
                and new_worked + new_lunch == worked + lunch
            ):
                row["worked"] = new_worked
                row["lunch"] = new_lunch
            else:
                raise ValueError(
                    "Invalid calculated lunch/worked totals"
                )

        rows.append(row)

    result["daily_timesheets"] = rows
    return result
