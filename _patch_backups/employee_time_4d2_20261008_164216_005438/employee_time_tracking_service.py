from datetime import datetime, timezone

from sqlalchemy import select
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

        if shift is None:
            shift = EmployeeShift(
                user_id=user.id,
                started_at=cls.now(),
                status="open",
                requires_review=False,
            )
            db.session.add(shift)
            db.session.flush()

        # Never clear a previous pending review.
        if not shift.requires_review:
            shift.status = "open"

        cls.record_event(shift, "LOGIN")
        db.session.commit()
        return shift


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
