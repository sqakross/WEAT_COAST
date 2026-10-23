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





        from sqlalchemy import update, select





        shift = cls.recovery_candidate(user)


        if shift is None:


            raise TimeTrackingError(


                "No eligible shift to resume. Start a new shift instead."


            )





        auto_close_exists = db.session.execute(


            select(EmployeeTimeEvent.id)


            .where(


                EmployeeTimeEvent.shift_id == shift.id,


                EmployeeTimeEvent.event_type == "AUTO_CLOSE_MIDNIGHT",


            )


            .limit(1)


        ).scalar_one_or_none()





        if auto_close_exists is not None:


            raise TimeTrackingError(


                "Automatically closed shift requires admin review."


            )





        old_end = shift.ended_at





        try:


            result = db.session.execute(


                update(EmployeeShift)


                .where(


                    EmployeeShift.id == shift.id,


                    EmployeeShift.user_id == user.id,


                    EmployeeShift.ended_at.is_not(None),


                    EmployeeShift.requires_review.is_(False),


                    EmployeeShift.status == "closed",


                )


                .values(


                    ended_at=None,


                    status="open",


                )


                .execution_options(synchronize_session=False)


            )





            if result.rowcount != 1:


                db.session.rollback()


                raise TimeTrackingError(


                    "Shift state changed. Refresh My Time."


                )





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


            db.session.refresh(shift)


            return shift





        except Exception:


            db.session.rollback()


            raise








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
    def _wccr_v22i_check_login_day(cls, shift):
        from zoneinfo import ZoneInfo
        la = ZoneInfo("America/Los_Angeles")

        def as_utc(value):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)

        started_day = as_utc(shift.started_at).astimezone(la).date()
        current_day = as_utc(cls.now()).astimezone(la).date()

        if started_day != current_day:
            raise TimeTrackingError(
                "Previous-day shift is still open. "
                "Admin review required before starting a new shift."
            )

    @classmethod


    def login(cls, user):


        if not cls.eligible(user):


            return None





        shift = cls.get_open_shift(user.id)





        if shift is not None:
            # WCCR_ATTENDANCE_ATOMIC_V22
            try:
                shift = cls._lock_open_shift_for_pause(user.id)
                # WCCR_MIDNIGHT_LOGIN_RECOVERY_V1
                from services.employee_midnight_engine import close_expired_shift
                from zoneinfo import ZoneInfo
                from datetime import timezone
                _wccr_today = cls.now().astimezone(ZoneInfo('America/Los_Angeles')).date()
                _wccr_started = shift.started_at
                if _wccr_started.tzinfo is None:
                    _wccr_started = _wccr_started.replace(tzinfo=timezone.utc)
                if _wccr_started.astimezone(ZoneInfo('America/Los_Angeles')).date() < _wccr_today:
                    _wccr_shift_id = shift.id
                    db.session.rollback()
                    if not close_expired_shift(_wccr_shift_id, now=cls.now()):
                        cls._wccr_v22i_check_login_day(shift)
                    return cls.login(user)
                cls._wccr_v22i_check_login_day(shift)
                if not shift.requires_review:
                    shift.status = "open"
                # WCCR_MULTI_BROWSER_LOGIN_V1
                # Existing shift: no duplicate LOGIN event.
                db.session.commit()
                return shift
            except TimeTrackingError as exc:
                if str(exc) not in (
                    "Shift already closed. Refresh My Time.",
                    "No open shift",
                ):
                    db.session.rollback()
                    raise

                db.session.rollback()
                previous = db.session.get(EmployeeShift, shift.id)
                if previous is None or previous.ended_at is None:
                    raise

                # WCCR_ATTENDANCE_ATOMIC_V22G
                # The previously selected shift was closed concurrently.
                # Continue through the original new-shift creation logic.
                db.session.expire_all()

            except Exception:
                db.session.rollback()
                raise

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





            # WCCR_MIDNIGHT_LOGIN_RECOVERY_V1
            from services.employee_midnight_engine import close_expired_shift
            from zoneinfo import ZoneInfo
            from datetime import timezone
            _wccr_today = cls.now().astimezone(ZoneInfo('America/Los_Angeles')).date()
            _wccr_started = existing.started_at
            if _wccr_started.tzinfo is None:
                _wccr_started = _wccr_started.replace(tzinfo=timezone.utc)
            if _wccr_started.astimezone(ZoneInfo('America/Los_Angeles')).date() < _wccr_today:
                _wccr_shift_id = existing.id
                db.session.rollback()
                if not close_expired_shift(_wccr_shift_id, now=cls.now()):
                    cls._wccr_v22i_check_login_day(existing)
                return cls.login(user)
            cls._wccr_v22i_check_login_day(existing)
            # WCCR_MULTI_BROWSER_LOGIN_V1
            # Concurrent login: keep existing shift.


            db.session.commit()


            return existing









    @classmethod


    def logout(cls, user):


        if not cls.eligible(user):


            return None





        # WCCR_ATTENDANCE_ATOMIC_V22
        try:
            shift = cls.get_open_shift(user.id)
            if shift is None:
                return None
            try:
                shift = cls._lock_open_shift_for_pause(user.id)
            except TimeTrackingError as exc:
                if str(exc) != "Shift already closed. Refresh My Time.":
                    raise

                db.session.rollback()

                closed_shift = db.session.get(EmployeeShift, shift.id)
                if closed_shift is None or closed_shift.ended_at is None:
                    raise

                # WCCR_ATTENDANCE_ATOMIC_V22E:
                # Another request has already closed this shift.
                # Do not add a late LOGOUT event.
                return closed_shift
            last_event = db.session.execute(
                select(EmployeeTimeEvent)
                .where(EmployeeTimeEvent.shift_id == shift.id)
                .order_by(EmployeeTimeEvent.id.desc())
            ).scalars().first()
            if last_event and last_event.event_type == "LOGOUT":
                db.session.rollback()
                return shift
            shift.requires_review = True
            shift.status = "pending_review"
            cls.record_event(
                shift, "LOGOUT",
                "Logout without End Shift; admin review required"
            )
            db.session.commit()
            return shift
        except Exception:
            db.session.rollback()
            raise








    @classmethod

    def end_shift(cls, user, expected_shift_id=None):

        if not cls.eligible(user):

            return None



        from sqlalchemy import update, case, or_



        try:

            if expected_shift_id is not None:
                current_shift = cls.get_open_shift(user.id)
                if (
                    current_shift is None
                    or current_shift.id != expected_shift_id
                ):
                    return None

            shift = cls._lock_open_shift_for_pause(user.id)

            # WCCR_MULTI_BROWSER_LOGOUT_V2
            # Verify identity while holding the shift write lock.
            if (
                expected_shift_id is not None
                and shift.id != expected_shift_id
            ):
                db.session.rollback()
                return None



            # The shift row is now write-locked.

            # Read pause events only after acquiring the lock.

            active_pause = cls.get_active_pause(shift)

            now = cls.now()



            review_expr = case(

                (

                    or_(

                        EmployeeShift.requires_review.is_(True),

                        active_pause is not None,

                    ),

                    True,

                ),

                else_=False,

            )



            status_expr = case(

                (

                    or_(

                        EmployeeShift.requires_review.is_(True),

                        active_pause is not None,

                    ),

                    "pending_review",

                ),

                else_="closed",

            )



            result = db.session.execute(

                update(EmployeeShift)

                .where(

                    EmployeeShift.id == shift.id,

                    EmployeeShift.user_id == user.id,

                    EmployeeShift.ended_at.is_(None),

                )

                .values(

                    ended_at=now,

                    status=status_expr,

                    requires_review=review_expr,

                )

                .execution_options(synchronize_session=False)

            )



            if result.rowcount != 1:

                raise TimeTrackingError(

                    "Shift already closed. Refresh My Time."

                )



            if active_pause is not None:

                cls.record_event(

                    shift,

                    "PAUSE_UNCLOSED",

                    f"Unclosed {active_pause} at End Shift",

                )



            cls.record_event(shift, "END_SHIFT")

            db.session.commit()

            db.session.refresh(shift)

            return shift



        except Exception:

            db.session.rollback()

            raise















    @classmethod


    def _lock_open_shift_for_pause(cls, user_id):


        from sqlalchemy import update





        shift = cls.get_open_shift(user_id)


        if shift is None:


            raise TimeTrackingError("No open shift")





        result = db.session.execute(


            update(EmployeeShift)


            .where(


                EmployeeShift.id == shift.id,


                EmployeeShift.user_id == user_id,


                EmployeeShift.ended_at.is_(None),


            )


            .values(status=EmployeeShift.status)


            .execution_options(synchronize_session=False)


        )





        if result.rowcount != 1:


            db.session.rollback()


            raise TimeTrackingError(


                "Shift already closed. Refresh My Time."


            )





        return shift





    @classmethod


    def start_pause(cls, user, kind):


        if not cls.eligible(user):


            return None





        if kind not in ("LUNCH", "BREAK"):


            raise TimeTrackingError("Invalid pause type")





        try:


            shift = cls._lock_open_shift_for_pause(user.id)





            active = cls.get_active_pause(shift)


            if active is not None:


                raise TimeTrackingError(


                    "Another lunch or break is already active"


                )





            cls.record_event(shift, f"{kind}_START")


            db.session.commit()


            return shift





        except Exception:


            db.session.rollback()


            raise








    @classmethod


    def end_pause(cls, user, kind):


        if not cls.eligible(user):


            return None





        if kind not in ("LUNCH", "BREAK"):


            raise TimeTrackingError("Invalid pause type")





        try:


            shift = cls._lock_open_shift_for_pause(user.id)





            active = cls.get_active_pause(shift)


            if active != kind:


                raise TimeTrackingError(


                    f"No active {kind.lower()} to end"


                )





            cls.record_event(shift, f"{kind}_END")


            db.session.commit()


            return shift





        except Exception:


            db.session.rollback()


            raise











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





        # WCCR_ATTENDANCE_ATOMIC_V22
        try:
            shift = cls.get_open_shift(user.id)
            if shift is None:
                return None
            shift = cls._lock_open_shift_for_pause(user.id)
            cls.record_event(shift, "IDLE")
            db.session.commit()
            return shift
        except TimeTrackingError as exc:
            if str(exc) != "Shift already closed. Refresh My Time.":
                db.session.rollback()
                raise

            db.session.rollback()
            current = db.session.get(EmployeeShift, shift.id)
            if current is None or current.ended_at is None:
                raise

            # WCCR_ATTENDANCE_ATOMIC_V22G
            # Do not record IDLE against a closed shift.
            return None

        except Exception:
            db.session.rollback()
            raise


