from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from extensions import db
from services.employee_time_tracking_models import (
    EmployeeShift,
    EmployeeTimeEvent,
    EmployeeTimeAudit,
)


class TimeTrackingError(Exception):
    pass


class EmployeeTimeTrackingService:

    EXCLUDED_ROLES = {"technician"}

    @staticmethod
    def now():
        return datetime.now(timezone.utc)

    @classmethod
    def eligible(cls, user):
        return (
            user is not None
            and getattr(user, "role", None) not in cls.EXCLUDED_ROLES
            and getattr(
                user, "employee_time_tracking_enabled", False
            ) is True
        )

    @staticmethod
    def get_open_shift(user_id):
        return db.session.execute(
            select(EmployeeShift)
            .where(
                EmployeeShift.user_id == user_id,
                EmployeeShift.ended_at.is_(None),
            )
            .order_by(EmployeeShift.id.desc())
        ).scalars().first()

    # WCCR_SMART_SHIFT_RECOVERY_4A
    @classmethod
    def recovery_candidate(cls, user):
        if not cls.eligible(user):
            return None
        from services.employee_time_policy import load_policy

        if not load_policy()["smart_shift_recovery"]:
            return None
        if cls.get_open_shift(user.id) is not None:
            return None

        shift = db.session.execute(
            select(EmployeeShift)
            .where(EmployeeShift.user_id == user.id)
            .order_by(EmployeeShift.started_at.desc(), EmployeeShift.id.desc())
            .limit(1)
        ).scalars().first()

        if shift is None or shift.ended_at is None:
            return None
        if shift.requires_review or shift.status != "closed":
            return None

        def aware(value):
            return (
                value.replace(tzinfo=timezone.utc)
                if value.tzinfo is None
                else value.astimezone(timezone.utc)
            )

        now = cls.now()
        ended = aware(shift.ended_at)
        tz = ZoneInfo("America/Los_Angeles")

        if ended > now:
            return None
        if ended.astimezone(tz).date() != now.astimezone(tz).date():
            return None

        events = db.session.execute(
            select(EmployeeTimeEvent)
            .where(EmployeeTimeEvent.shift_id == shift.id)
            .order_by(EmployeeTimeEvent.id)
        ).scalars().all()

        # A shift with an unclosed pause must be reviewed,
        # not automatically recovered.
        active = None
        for event in events:
            if event.event_type in ("LUNCH_START", "BREAK_START"):
                active = event.event_type.split("_")[0]
            elif event.event_type in ("LUNCH_END", "BREAK_END"):
                active = None
        if active is not None:
            return None

        if not events or events[-1].event_type != "END_SHIFT":
            return None

        return shift

    @classmethod
    def resume_shift(cls, user):
        if not cls.eligible(user):
            raise TimeTrackingError("Time tracking is not enabled")

        shift = cls.recovery_candidate(user)
        if shift is None:
            raise TimeTrackingError(
                "No eligible shift to resume. Start a new shift instead."
            )

        old_end = shift.ended_at
        now = cls.now()
        shift.ended_at = None
        shift.status = "open"

        cls.record_event(
            shift,
            "RESUME_SHIFT",
            "Explicit recovery after End Shift; gap is unpaid",
        )
        db.session.add(EmployeeTimeAudit(
            shift_id=shift.id,
            changed_by_user_id=user.id,
            field_name="ended_at",
            old_value=str(old_end),
            new_value="None",
            reason="Employee explicitly resumed a closed shift",
        ))
        db.session.commit()
        return shift

    @classmethod
    def record_event(cls, shift, event_type, note=None):
        event = EmployeeTimeEvent(
            shift_id=shift.id,
            user_id=shift.user_id,
            event_type=event_type,
            occurred_at=cls.now(),
            note=note,
        )
        db.session.add(event)
        return event


    @classmethod
    def login(cls, user):
        if not cls.eligible(user):
            return None

        shift = cls.get_open_shift(user.id)

        if shift is not None:
            if not shift.requires_review:
                shift.status = "open"

            cls.record_event(shift, "LOGIN")
            db.session.commit()
            return shift

        # WCCR_MULTIPLE_INTERVALS_V1_1: completed shifts do not block Start

        shift = EmployeeShift(
            user_id=user.id,
            started_at=cls.now(),
            status="open",
            requires_review=False,
        )

        try:
            db.session.add(shift)
            db.session.flush()

            cls.record_event(shift, "LOGIN")
            db.session.commit()
            return shift

        except IntegrityError as exc:
            db.session.rollback()

            # Only recover from the known open-shift unique index.
            message = str(exc.orig).lower()

            if (
                "unique constraint failed" not in message
                or "employee_time_shifts.user_id" not in message
            ):
                raise

            existing = cls.get_open_shift(user.id)

            if existing is None:
                raise

            if not existing.requires_review:
                existing.status = "open"

            cls.record_event(existing, "LOGIN")
            db.session.commit()
            return existing


    @classmethod
    def logout(cls, user):
        if not cls.eligible(user):
            return None

        shift = cls.get_open_shift(user.id)
        if shift is None:
            return None

        last_event = db.session.execute(
            select(EmployeeTimeEvent)
            .where(EmployeeTimeEvent.shift_id == shift.id)
            .order_by(EmployeeTimeEvent.id.desc())
        ).scalars().first()

        if last_event and last_event.event_type == "LOGOUT":
            return shift

        shift.requires_review = True
        shift.status = "pending_review"

        cls.record_event(
            shift,
            "LOGOUT",
            "Logout without End Shift; admin review required",
        )

        db.session.commit()
        return shift


    @classmethod
    def end_shift(cls, user):
        if not cls.eligible(user):
            return None

        shift = cls.get_open_shift(user.id)

        if shift is None:
            raise TimeTrackingError("No open shift")

        active_pause = cls.get_active_pause(shift)

        if active_pause is not None:
            shift.requires_review = True

            cls.record_event(
                shift,
                "PAUSE_UNCLOSED",
                f"Unclosed {active_pause} at End Shift",
            )

        shift.ended_at = cls.now()
        shift.status = (
            "pending_review"
            if shift.requires_review
            else "closed"
        )

        cls.record_event(shift, "END_SHIFT")
        db.session.commit()
        return shift

    @classmethod
    def start_pause(cls, user, kind):
        if not cls.eligible(user):
            return None

        if kind not in ("LUNCH", "BREAK"):
            raise TimeTrackingError("Invalid pause type")

        shift = cls.get_open_shift(user.id)

        if shift is None:
            raise TimeTrackingError("No open shift")

        events = db.session.execute(
            select(EmployeeTimeEvent)
            .where(EmployeeTimeEvent.shift_id == shift.id)
            .order_by(EmployeeTimeEvent.id)
        ).scalars().all()

        active = None

        for event in events:
            if event.event_type in ("LUNCH_START", "BREAK_START"):
                active = event.event_type.split("_")[0]
            elif event.event_type in ("LUNCH_END", "BREAK_END"):
                active = None

        if active:
            raise TimeTrackingError(
                "Another lunch or break is already active"
            )

        cls.record_event(shift, f"{kind}_START")
        db.session.commit()
        return shift

    @classmethod
    def end_pause(cls, user, kind):
        if not cls.eligible(user):
            return None

        if kind not in ("LUNCH", "BREAK"):
            raise TimeTrackingError("Invalid pause type")

        shift = cls.get_open_shift(user.id)

        if shift is None:
            raise TimeTrackingError("No open shift")

        events = db.session.execute(
            select(EmployeeTimeEvent)
            .where(EmployeeTimeEvent.shift_id == shift.id)
            .order_by(EmployeeTimeEvent.id)
        ).scalars().all()

        active = None

        for event in events:
            if event.event_type in ("LUNCH_START", "BREAK_START"):
                active = event.event_type.split("_")[0]
            elif event.event_type in ("LUNCH_END", "BREAK_END"):
                active = None

        if active != kind:
            raise TimeTrackingError(
                f"No active {kind.lower()} to end"
            )

        cls.record_event(shift, f"{kind}_END")
        db.session.commit()
        return shift


    @staticmethod
    def get_active_pause(shift):
        events = db.session.execute(
            select(EmployeeTimeEvent)
            .where(EmployeeTimeEvent.shift_id == shift.id)
            .order_by(EmployeeTimeEvent.id)
        ).scalars().all()

        active = None

        for event in events:
            if event.event_type in ("LUNCH_START", "BREAK_START"):
                active = event.event_type.split("_")[0]
            elif event.event_type in ("LUNCH_END", "BREAK_END"):
                active = None

        return active

    @classmethod
    def record_idle(cls, user):
        if not cls.eligible(user):
            return None

        shift = cls.get_open_shift(user.id)

        if shift is None:
            return None

        cls.record_event(shift, "IDLE")
        db.session.commit()
        return shift
