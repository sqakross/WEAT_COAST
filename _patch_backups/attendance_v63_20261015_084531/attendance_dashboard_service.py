from collections import defaultdict
from datetime import datetime, timedelta, timezone, time
from zoneinfo import ZoneInfo
from sqlalchemy import select, or_
from models import User

from services.employee_time_tracking_models import (
    EmployeeShift, EmployeeTimeEvent
)
from services.employee_time_calculations import (
    make_time_calculator
)


def get_attendance_dashboard_data(
    db, *, paid_break_minutes, can_review=False,
    now=None, days=14, start_date=None, end_date=None
):
    if now is None:
        now = datetime.now(timezone.utc)

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")

    if type(days) is not int or not 1 <= days <= 31:
        raise ValueError("days must be between 1 and 31")

    tz = ZoneInfo("America/Los_Angeles")
    today = now.astimezone(tz).date()
    first_day = today - timedelta(days=days - 1)
    # WCCR_ACCOUNTING_TIMESHEET_V61
    actual_today = today
    if start_date is not None or end_date is not None:
        from datetime import date as _date
        if (
            type(start_date) is not _date
            or type(end_date) is not _date
            or end_date < start_date
            or end_date > actual_today
            or (end_date - start_date).days > 30
        ):
            raise ValueError("Invalid selected timesheet period")
        first_day = start_date
        today = end_date

    def midnight(day):
        return datetime.combine(
            day, time.min, tzinfo=tz
        ).astimezone(timezone.utc)

    left = midnight(first_day)
    selected_end = min(
        now, midnight(today + timedelta(days=1))
    )

    (
        aware, la_time, seconds_between, summary,
        overlap, totals, utc_midnight
    ) = make_time_calculator(
        paid_break_minutes, tz
    )

    shifts = db.session.execute(
        select(EmployeeShift)
        .where(
            EmployeeShift.started_at < selected_end,
            or_(
                EmployeeShift.ended_at.is_(None),
                EmployeeShift.ended_at > left
            )
        )
        .order_by(EmployeeShift.started_at.desc())
    ).scalars().all()

    events_by_shift = defaultdict(list)
    ids = [s.id for s in shifts]

    user_ids = sorted({s.user_id for s in shifts})
    usernames = {}
    if user_ids:
        user_rows = db.session.execute(
            select(User.id, User.username)
            .where(User.id.in_(user_ids))
        ).all()
        usernames = {
            user_id: username
            for user_id, username in user_rows
        }

    if ids:
        events = db.session.execute(
            select(EmployeeTimeEvent)
            .where(EmployeeTimeEvent.shift_id.in_(ids))
            .order_by(
                EmployeeTimeEvent.shift_id,
                EmployeeTimeEvent.id
            )
        ).scalars().all()

        for event in events:
            events_by_shift[event.shift_id].append(event)

    # WCCR_ATTENDANCE_V62
    # Repeated LOGIN without LOGOUT is ambiguous.
    # Do not silently reinterpret historical attendance events.
    ambiguous_shift_ids = set()
    for _shift in shifts:
        _logged_in = False
        for _event in events_by_shift[_shift.id]:
            _kind = _event.event_type
            if _kind == "LOGIN":
                if _logged_in:
                    ambiguous_shift_ids.add(_shift.id)
                _logged_in = True
            elif _kind in ("LOGOUT", "END_SHIFT"):
                _logged_in = False

    working = []
    review = []
    overnight = []
    timesheets = []
    today_totals = defaultdict(int)

    for shift in shifts:
        info = summary(
            shift, events_by_shift[shift.id], now
        )

        open_shift = shift.ended_at is None
        requires_review = bool(
            shift.requires_review
            or info["anomalies"]
            or shift.id in ambiguous_shift_ids
        )

        elapsed = (
            max(
                0,
                (
                    now - aware(shift.started_at)
                ).total_seconds() / 3600
            )
            if open_shift else 0
        )

        record = {
            "shift_id": shift.id,
            "user_id": shift.user_id,
            "employee_name": (
                usernames.get(shift.user_id)
                or f"User #{shift.user_id}"
            ),
            "started": la_time(shift.started_at),
            "ended": (
                la_time(shift.ended_at)
                if shift.ended_at else None
            ),
            "status": shift.status,
            "open": open_shift,
            "active_pause": info["active_pause"],
            "requires_review": requires_review if can_review else False,
            "has_idle": info["has_idle"],
            "elapsed_hours": round(elapsed, 2),
        }

        if open_shift:
            working.append(record)

        if requires_review and can_review:
            review.append(record)

        if open_shift and elapsed > 12:
            overnight.append(record)

        day = first_day
        while day <= today:
            start = midnight(day)
            finish = min(
                selected_end,
                midnight(day + timedelta(days=1))
            )
            amounts = totals(info, start, finish)

            if (
                amounts["total"] > 0
                or (
                    requires_review
                    and info["start"] < finish
                    and info["end"] > start
                )
            ):
                timesheets.append({
                    **record,
                    "date": day.isoformat(),
                    "unverified": (
                        shift.id in ambiguous_shift_ids
                    ),
                    "daily_start": max(info["start"], start),
                    "daily_end": min(info["end"], finish),
                    **amounts,
                })

                if day == today:
                    today_totals[shift.user_id] += (
                        amounts["worked"]
                    )

            day += timedelta(days=1)


    # V6.1 - aggregate per employee and California calendar day.
    daily_groups = defaultdict(list)
    for part in timesheets:
        daily_groups[(part["user_id"], part["date"])].append(part)

    daily_timesheets = []
    for (employee_id, work_date), parts in daily_groups.items():
        parts.sort(key=lambda x: (
            x["daily_start"], x["daily_end"], x["shift_id"]
        ))

        max_end = None
        overlapping = False
        for part in parts:
            if max_end is not None and part["daily_start"] < max_end:
                overlapping = True
            if max_end is None or part["daily_end"] > max_end:
                max_end = part["daily_end"]

        unverified = any(
            p.get("unverified", False) for p in parts
        )
        review_required = (
            overlapping
            or unverified
            or any(p["requires_review"] for p in parts)
        )
        is_open = any(p["open"] for p in parts)
        first = parts[0]

        daily_timesheets.append({
            "date": work_date,
            "user_id": employee_id,
            "employee_name": first["employee_name"],
            "started": first["daily_start"].astimezone(
                tz
            ).strftime("%I:%M %p"),
            "ended": (
                "Still open" if is_open else
                max_end.astimezone(tz).strftime("%I:%M %p")
            ),
            "unverified": unverified,
            "worked": (
                None if overlapping or unverified else
                sum(p["worked"] for p in parts)
            ),
            "lunch": (
                None if overlapping else
                sum(p["lunch"] for p in parts)
            ),
            "break": (
                None if overlapping else
                sum(p["break"] for p in parts)
            ),
            "open": is_open,
            "needs_review": review_required,
            "overlap": overlapping,
            "parts": parts,
        })

    daily_timesheets.sort(key=lambda x: (
        x["date"], x["user_id"]
    ), reverse=True)

    timesheets.sort(
        key=lambda row: (
            row["date"],
            row["user_id"],
            row["shift_id"]
        ),
        reverse=True
    )

    return {
        "working": working,
        "pending_review": review,
        "overnight": overnight,
        "timesheets": timesheets,
        "daily_timesheets": daily_timesheets,
        "today_totals": dict(today_totals),
        "today": today.isoformat(),
        "first_day": first_day.isoformat(),
        "preliminary": True,
    }
