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

# WCCR_ATTENDANCE_V73G_APPROVED_EFFECTIVE
from services.attendance_review_v73 import (
    AttendanceReviewRevision,
    AttendanceReviewPeriod,
    aware_utc,
    validate_revision,
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


    # V7.3G: load saved revisions in bulk.
    # Never mutate original EmployeeShift / EmployeeTimeEvent.
    approved_effective = {}

    if ids:
        _review_revisions = db.session.execute(
            select(AttendanceReviewRevision)
            .where(
                AttendanceReviewRevision.shift_id.in_(ids)
            )
            .order_by(
                AttendanceReviewRevision.shift_id,
                AttendanceReviewRevision.revision.desc(),
                AttendanceReviewRevision.id.desc()
            )
        ).scalars().all()

        _latest_by_shift = {}

        for _revision in _review_revisions:
            _latest_by_shift.setdefault(
                _revision.shift_id, _revision
            )

        _approved_candidates = {
            shift_id: revision
            for shift_id, revision in _latest_by_shift.items()
            if revision.status == "approved"
        }

        if _approved_candidates:
            _period_rows = db.session.execute(
                select(AttendanceReviewPeriod)
                .where(
                    AttendanceReviewPeriod.revision_id.in_(
                        [
                            revision.id
                            for revision
                            in _approved_candidates.values()
                        ]
                    )
                )
                .order_by(
                    AttendanceReviewPeriod.revision_id,
                    AttendanceReviewPeriod.started_at_utc,
                    AttendanceReviewPeriod.id
                )
            ).scalars().all()

            _periods_by_revision = defaultdict(list)

            for _period in _period_rows:
                _periods_by_revision[
                    _period.revision_id
                ].append(_period)

            for _shift in shifts:
                _revision = _approved_candidates.get(_shift.id)

                if _revision is None:
                    continue

                if _shift.ended_at is None:
                    continue

                # Special event sequences may represent
                # recovery gaps or disabled tracking.
                # Do not discard those accounting rules.
                _source_kinds = {
                    event.event_type
                    for event in events_by_shift[_shift.id]
                }

                # WCCR_V73H5_APPROVED_END_SHIFT
                # END_SHIFT is a normal explicit shift-closing event.
                # For an approved revision, validation below determines
                # whether its corrected times can be applied.
                # Keep tracking/recovery/idle/logout exclusions.
                _unsafe_kinds = {
                    "TRACKING_DISABLED",
                    "RESUME_SHIFT",
                    "IDLE",
                    "LOGOUT",
                }

                if _source_kinds & _unsafe_kinds:
                    continue

                if not _source_kinds.issubset({
                    "LOGIN",
                    "END_SHIFT",
                    "AUTO_CLOSE_MIDNIGHT",
                    "LUNCH_START",
                    "LUNCH_END",
                    "BREAK_START",
                    "BREAK_END",
                }):
                    continue

                _saved_intervals = []

                for _item in _periods_by_revision.get(
                    _revision.id, []
                ):
                    _saved_intervals.append({
                        "kind": _item.kind,
                        "start": aware_utc(
                            _item.started_at_utc
                        ),
                        "end": aware_utc(
                            _item.ended_at_utc
                        ),
                    })

                try:
                    _validated = validate_revision(
                        _revision.started_at_utc,
                        _revision.ended_at_utc,
                        _saved_intervals,
                        _revision.reason,
                        now=max(
                            now,
                            aware_utc(
                                _revision.ended_at_utc
                            )
                        ),
                    )
                except (ValueError, TypeError):
                    continue

                approved_effective[_shift.id] = _validated

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

    # WCCR_ATTENDANCE_V63
    # Reporting-only safeguards. Never alter recorded events.
    unverified_shift_ids = set(ambiguous_shift_ids)

    for _shift in shifts:
        _shift_start = aware(_shift.started_at)
        _shift_end = aware(_shift.ended_at) or now

        if (_shift_end - _shift_start).total_seconds() > 16 * 3600:
            unverified_shift_ids.add(_shift.id)

        if any(
            _event.event_type in (
                "TRACKING_DISABLED",
                "AUTO_CLOSE_MIDNIGHT",
            )
            for _event in events_by_shift[_shift.id]
        ):
            unverified_shift_ids.add(_shift.id)

    working = []
    review = []
    overnight = []
    timesheets = []
    today_totals = defaultdict(int)

    for shift in shifts:
        info = summary(
            shift, events_by_shift[shift.id], now
        )

        # V7.3G: approved correction replaces the effective
        # calculation, not the original recorded events.
        _effective = approved_effective.get(shift.id)

        if _effective is not None:
            info = {
                "start": _effective["start"],
                "end": _effective["end"],
                "intervals": {
                    "LUNCH": [
                        (p["start"], p["end"])
                        for p in _effective["periods"]
                        if p["kind"] == "Lunch"
                    ],
                    "BREAK": [
                        (p["start"], p["end"])
                        for p in _effective["periods"]
                        if p["kind"] == "Break"
                    ],
                },
                "recovery_gaps": [],
                "logout_gaps": [],
                "active_pause": None,
                "anomalies": False,
                "has_idle": False,
            }

        open_shift = shift.ended_at is None
        requires_review = bool(
            shift.requires_review
            or info["anomalies"]
            or shift.id in unverified_shift_ids
        )

        if _effective is not None:
            # Explicit Attendance approval resolves the
            # original auto-close review requirement.
            requires_review = False
            unverified_shift_ids.discard(shift.id)


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
            "started": la_time(info["start"]),
            "ended": (
                la_time(info["end"])
                if not open_shift else None
            ),
            "status": (
                "approved"
                if _effective is not None
                else shift.status
            ),
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
                        shift.id in unverified_shift_ids
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

    # WCCR_TODAYS_WORKED_PRELIMINARY_V64
    # Include calculated hours even when review is pending.
    # Exclude unresolved/invalid daily totals.
    # This is display-only, not payroll approval.
    today_totals = defaultdict(int)
    for _day_row in daily_timesheets:
        if (
            _day_row["date"] == actual_today.isoformat()
            and _day_row["worked"] is not None
        ):
            today_totals[_day_row["user_id"]] += (
                _day_row["worked"]
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
