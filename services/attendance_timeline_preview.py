
"""Read-only, conservative attendance timeline preview.

Not a payroll calculation or an approval engine.
Original events are never modified.
"""

from datetime import timezone
from zoneinfo import ZoneInfo

PACIFIC = ZoneInfo("America/Los_Angeles")


def reconstruct_preview(events):
    ordered = sorted(
        events,
        key=lambda e: (e.occurred_at, e.id)
    )

    def instant(e):
        t = e.occurred_at
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return t.astimezone(timezone.utc)

    def display(t):
        return t.astimezone(PACIFIC).strftime(
            "%b %d, %Y %I:%M %p"
        )

    rows = []
    warnings = []
    active_work = None
    active_pause = None
    uncertain_work = False

    def add(kind, start, end=None, status="Review Required",
            detail="", ids=None):
        rows.append({
            "type": kind,
            "start": display(instant(start)) if start else "?",
            "end": display(instant(end)) if end else "?",
            "status": status,
            "detail": detail,
            "event_ids": ", ".join(
                str(i) for i in (ids or [])
            )
        })

    for e in ordered:
        kind = e.event_type
        time = instant(e)

        if kind == "LOGIN":
            if active_work is not None:
                add(
                    "Work candidate", active_work, e,
                    detail="Repeated LOGIN; work boundary uncertain",
                    ids=[active_work.id, e.id]
                )
                warnings.append("Repeated LOGIN")
                uncertain_work = True
            active_work = e

        elif kind == "END_SHIFT":
            if active_work is not None:
                add(
                    "Work candidate", active_work, e,
                    status=(
                        "Review Required" if uncertain_work
                        or active_pause is not None
                        else "Candidate"
                    ),
                    detail=(
                        "Not verified against recovery/payroll engine"
                    ),
                    ids=[active_work.id, e.id]
                )
                active_work = None
                uncertain_work = False
            else:
                add(
                    "Unmatched END_SHIFT", e,
                    detail="No unambiguous active work start",
                    ids=[e.id]
                )
                warnings.append("Unmatched END_SHIFT")

        elif kind == "RESUME_SHIFT":
            add(
                "Recovery / Resume", e,
                detail="Recovery boundary requires review",
                ids=[e.id]
            )
            warnings.append("Recovery event")
            uncertain_work = True

        elif kind in ("LUNCH_START", "BREAK_START"):
            if active_pause is not None:
                add(
                    "Incomplete " + active_pause.event_type,
                    active_pause,
                    detail="Another pause began before this one ended",
                    ids=[active_pause.id]
                )
                warnings.append("Overlapping pause")
            active_pause = e

        elif kind in ("LUNCH_END", "BREAK_END"):
            expected = (
                "LUNCH_START" if kind == "LUNCH_END"
                else "BREAK_START"
            )
            if active_pause and active_pause.event_type == expected:
                add(
                    "Lunch" if kind == "LUNCH_END" else "Break",
                    active_pause, e,
                    status="Candidate",
                    detail="Paired original events; payroll not verified",
                    ids=[active_pause.id, e.id]
                )
                active_pause = None
            else:
                add(
                    "Unmatched " + kind, e,
                    detail="Missing matching pause start",
                    ids=[e.id]
                )
                warnings.append("Unmatched pause end")

        else:
            add(
                kind, e,
                detail="Event requires manual classification",
                ids=[e.id]
            )
            warnings.append("Unclassified event")

    if active_work:
        add(
            "Open work candidate", active_work,
            detail="No matching END_SHIFT",
            ids=[active_work.id]
        )
        warnings.append("Unclosed work interval")

    if active_pause:
        add(
            "Incomplete pause", active_pause,
            detail="Missing matching pause end",
            ids=[active_pause.id]
        )
        warnings.append("Unclosed pause")

    # Same-time recovery events can create false intervals.
    seen = {}
    for e in ordered:
        key = (e.event_type, instant(e))
        if key in seen:
            warnings.append(
                "Duplicate event type/time: "
                + str(seen[key]) + ", " + str(e.id)
            )
        else:
            seen[key] = e.id

    return {
        "rows": rows,
        "warnings": sorted(set(warnings)),
        "event_count": len(ordered),
        "verified_payroll": False
    }
