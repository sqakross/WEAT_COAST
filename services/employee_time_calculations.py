"""Shared WCCR attendance calculations.
Extracted without changing calculation formulas.
Payroll approval must use separately validated rules.
"""
from datetime import datetime, time, timezone

def make_time_calculator(paid_break_limit, tz):
    def aware(value):
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def la_time(value):
        value = aware(value)
        return (
            value.astimezone(tz).strftime("%b %d, %Y %I:%M %p")
            if value else "?"
        )

    def seconds_between(start, end):
        return max(0, (end - start).total_seconds())

    # WCCR_SMART_SHIFT_RECOVERY_4A
    def summary(shift, events, end_time):
        start = aware(shift.started_at)
        finish = aware(shift.ended_at) or end_time
        if finish < start:
            finish = start

        intervals = {"LUNCH": [], "BREAK": []}
        recovery_gaps = []
        gap_start = None
        # WCCR_LOGOUT_GAP_V13
        logout_gaps = []
        logout_start = None
        active_kind = None
        active_start = None
        anomalies = False
        has_idle = False

        for event in events:
            kind = event.event_type
            at = aware(event.occurred_at)

            if kind == "IDLE":
                has_idle = True

            if at is None or at < start or at > finish:
                continue

            # WCCR_LOGOUT_GAP_V13
            if kind == "LOGOUT":
                if logout_start is None:
                    logout_start = at
                else:
                    anomalies = True
            elif kind == "LOGIN":
                if logout_start is not None:
                    logout_gaps.append((logout_start, at))
                    logout_start = None

            if kind == "END_SHIFT":
                if gap_start is None:
                    gap_start = at
                else:
                    anomalies = True
            elif kind == "RESUME_SHIFT":
                if gap_start is not None:
                    recovery_gaps.append((gap_start, at))
                    gap_start = None
                else:
                    anomalies = True

            if kind in ("LUNCH_START", "BREAK_START"):
                next_kind = kind.split("_")[0]
                if active_kind is not None:
                    anomalies = True
                else:
                    active_kind = next_kind
                    active_start = at

            elif kind in ("LUNCH_END", "BREAK_END"):
                ending_kind = kind.split("_")[0]
                if active_kind == ending_kind:
                    intervals[ending_kind].append(
                        (active_start, at)
                    )
                    active_kind = None
                    active_start = None
                else:
                    anomalies = True

        if active_kind is not None:
            intervals[active_kind].append(
                (active_start, finish)
            )

        # WCCR_LOGOUT_GAP_V13
        if logout_start is not None:
            logout_gaps.append((logout_start, finish))

        return {
            "start": start,
            "end": finish,
            "intervals": intervals,
            "recovery_gaps": recovery_gaps,
            "logout_gaps": logout_gaps,
            "active_pause": active_kind,
            "anomalies": anomalies,
            "has_idle": has_idle,
        }

    def overlap(a, b, left, right):
        return seconds_between(max(a, left), min(b, right))

    # WCCR_PAID_BREAK_V1
    # WCCR_PAID_BREAK_FIELDS_V2A
    def totals(info, left, right):
        total = overlap(info["start"], info["end"], left, right)

        lunch = sum(
            overlap(a, b, left, right)
            for a, b in info["intervals"]["LUNCH"]
        )

        break_total = sum(
            overlap(a, b, left, right)
            for a, b in info["intervals"]["BREAK"]
        )

        # Paid allowance is per entire shift, not per report day.
        allowance = paid_break_limit * 60

        break_before = sum(
            overlap(a, b, info["start"], left)
            for a, b in info["intervals"]["BREAK"]
        )

        paid_before = min(allowance, break_before)
        paid_through = min(
            allowance,
            break_before + break_total
        )

        paid_break = max(0, paid_through - paid_before)
        unpaid_break = max(0, break_total - paid_break)

        lunch = min(total, lunch)
        unpaid_break = min(
            max(0, total - lunch),
            unpaid_break
        )

        # WCCR_SMART_SHIFT_RECOVERY_4A
        # Total Shift remains elapsed calendar time; recovery gaps
        # are excluded from Worked Time.
        recovery_gap = sum(
            overlap(a, b, left, right)
            for a, b in info["recovery_gaps"]
        )
        # WCCR_LOGOUT_GAP_V13
        # Union unpaid intervals to prevent double subtraction.
        unpaid_intervals = (
            info["intervals"]["LUNCH"]
            + info["recovery_gaps"]
            + info.get("logout_gaps", [])
        )

        clipped = []
        for a, b in unpaid_intervals:
            a = max(a, left, info["start"])
            b = min(b, right, info["end"])
            if b > a:
                clipped.append((a, b))

        clipped.sort(key=lambda pair: pair[0])
        merged = []

        for a, b in clipped:
            if merged and a <= merged[-1][1]:
                merged[-1] = (
                    merged[-1][0],
                    max(merged[-1][1], b)
                )
            else:
                merged.append((a, b))

        unpaid_union = sum(
            seconds_between(a, b)
            for a, b in merged
        )

        # Break allowance is retained; unpaid break portions
        # overlapping other unpaid gaps must not be subtracted twice.
        extra_unpaid_break = 0
        for a, b in info["intervals"]["BREAK"]:
            a = max(a, left, info["start"])
            b = min(b, right, info["end"])
            if b <= a:
                continue
            overlap_unpaid = sum(
                overlap(a, b, x, y)
                for x, y in merged
            )
            extra_unpaid_break += max(
                0,
                seconds_between(a, b) - overlap_unpaid
            )

        extra_unpaid_break = min(
            unpaid_break,
            extra_unpaid_break
        )

        worked = max(
            0,
            total - unpaid_union - extra_unpaid_break
        )


        # WCCR_MY_TIME_ACTUAL_WORK_V663
        excluded_intervals = (
            info["intervals"]["LUNCH"]
            + info["intervals"]["BREAK"]
            + info["recovery_gaps"]
            + info.get("logout_gaps", [])
        )

        actual_intervals = []
        for a, b in excluded_intervals:
            a = max(a, left, info["start"])
            b = min(b, right, info["end"])
            if b > a:
                actual_intervals.append((a, b))

        actual_intervals.sort(key=lambda item: item[0])
        actual_merged = []

        for a, b in actual_intervals:
            if actual_merged and a <= actual_merged[-1][1]:
                actual_merged[-1] = (
                    actual_merged[-1][0],
                    max(actual_merged[-1][1], b)
                )
            else:
                actual_merged.append((a, b))

        actual_work = max(
            0,
            total - sum(
                seconds_between(a, b)
                for a, b in actual_merged
            )
        )

        return {
            "total": int(total),
            "lunch": int(lunch),
            "break": int(break_total),
            "paid_break": int(paid_break),
            "unpaid_break": int(unpaid_break),
            "worked": int(worked),
            "actual_work": int(actual_work),
            "paid_time": int(worked),
        }

    def utc_midnight(day):
        return datetime.combine(
            day, time.min, tzinfo=tz
        ).astimezone(timezone.utc)
    return (aware, la_time, seconds_between, summary, overlap, totals, utc_midnight)
