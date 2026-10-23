# WCCR_MY_TIME_FILTERS_V68
# WCCR_MY_TIME_SIMPLE_V671
import time
from flask import Blueprint, render_template, redirect, url_for, flash, request, session
from flask_login import login_user, logout_user, login_required, current_user
from sqlalchemy import func
import logging
from services.employee_time_tracking_service import EmployeeTimeTrackingService
from sqlalchemy.orm import lazyload
from extensions import db, login_manager
from models import (
    User,
    ROLE_SUPERADMIN,
    ROLE_ADMIN,
    ROLE_ACCOUNTING,
    ROLE_USER,
    ROLE_VIEWER,
    ROLE_TECHNICIAN,
    ALLOWED_ROLES,
    ROLE_ALIASES,
)

auth_bp = Blueprint('auth', __name__)

@login_manager.user_loader
def load_user(user_id):
    try:
        uid = int(user_id)

        return (
            db.session.query(User)
            .options(lazyload("*"))
            .filter(User.id == uid)
            .first()
        )

    except (TypeError, ValueError):
        return None
# Login
@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter(func.lower(User.username) == username.lower()).first()
        if user and user.check_password(password):
            login_user(user)
            # WCCR_MULTI_BROWSER_LOGOUT_V2
            session.pop("_wccr_auto_shift_id_v2", None)

            # WCCR_EMPLOYEE_TIME_LOGIN_START
            if (EmployeeTimeTrackingService.eligible(user) and user.employee_time_tracking_mode == 'automatic'):
                try:
                    _wccr_bound_shift = EmployeeTimeTrackingService.login(user)
                    if _wccr_bound_shift is not None:
                        session["_wccr_auto_shift_id_v2"] = int(
                            _wccr_bound_shift.id
                        )
                except Exception:
                    db.session.rollback()
                    logging.exception(
                        'Employee Time Tracking login failed: user_id=%s',
                        user.id,
                    )
            # WCCR_EMPLOYEE_TIME_LOGIN_END

            # WCCR_TECH_INACTIVITY_LOGIN_RESET
            if (
                (getattr(user, "role", "") or "")
                .strip()
                .lower()
                == "technician"
            ):
                session["tech_last_activity"] = time.time()
                session.modified = True
            else:
                session.pop("tech_last_activity", None)

            # WCCR APPLIANCE PERMISSION-AWARE LANDING V2 START
            next_url = request.args.get("next")

            # Preserve an explicit destination supplied by Flask-Login.
            if next_url:
                return redirect(next_url)

            # Use the existing authorization engines.
            #
            # Rule:
            #   Appliance access = YES
            #   Legacy Parts Inventory access = NO
            #       -> Appliance Inventory
            #
            # This is permission-based. No username or role is
            # hard-coded here.
            from services.access_control_service import (
                AccessControlService,
            )
            from services.erp_access_service import (
                ErpAccessService,
            )

            can_view_appliances = (
                AccessControlService.can(
                    user,
                    "appliance.view",
                )
            )

            can_access_parts_inventory = (
                ErpAccessService.is_allowed(
                    user,
                    "erp.inventory.access",
                    default_allowed=True,
                )
            )

            if (
                can_view_appliances
                and not can_access_parts_inventory
            ):
                return redirect(
                    url_for(
                        "appliance.inventory_list"
                    )
                )

            # Preserve the historical landing for everybody else.
            return redirect(
                url_for("inventory.wo_list")
            )
            # WCCR APPLIANCE PERMISSION-AWARE LANDING V2 END

        flash('Invalid username or password', 'danger')
    return render_template('login.html')


# Logout
@auth_bp.route('/logout')
@login_required
def logout():
    # WCCR_EMPLOYEE_TIME_LOGOUT_START
    if (EmployeeTimeTrackingService.eligible(current_user) and current_user.employee_time_tracking_mode == 'automatic'):
        try:
            # WCCR_CLOCK_V66A
            # Sign Out closes the current work interval.
            # WCCR_MULTI_BROWSER_LOGOUT_V2
            _wccr_bound_id = session.get("_wccr_auto_shift_id_v2")
            if _wccr_bound_id is not None:
                EmployeeTimeTrackingService.end_shift(
                    current_user,
                    expected_shift_id=int(_wccr_bound_id),
                )
            else:
                flash(
                    "Automatic Clock Out was not performed: "
                    "this browser session predates shift protection. "
                    "Please verify your time in My Time.",
                    "warning",
                )
        except Exception:
            db.session.rollback()
            logging.exception(
                'Employee Time Tracking logout failed: user_id=%s',
                current_user.id,
            )
    
            flash('Automatic Clock Out failed. Please contact your supervisor.', 'danger')
            return redirect(url_for('auth.my_time'))
# WCCR_EMPLOYEE_TIME_LOGOUT_END
    logout_user()
    session.pop("tech_last_activity", None)
    return redirect(url_for('auth.login'))

# Register new user (superadmin only)
@auth_bp.route('/register', methods=['GET', 'POST'])
@login_required
def register():
    if current_user.role != ROLE_SUPERADMIN:
        flash('Access denied', 'danger')
        return redirect(url_for('inventory.wo_list'))

    if request.method == 'POST':
        username = (request.form.get('username') or '').strip()
        password = request.form.get('password') or ''
        role_raw = (request.form.get('role') or '').strip().lower()
        role = ROLE_ALIASES.get(role_raw, role_raw)
        if role not in ALLOWED_ROLES:
            role = ROLE_TECHNICIAN  # дефолт/санитайз

        if not username or not password:
            flash('Username and password are required', 'danger')
            return redirect(url_for('auth.register'))

        if User.query.filter(func.lower(User.username) == username.lower()).first():
            flash('User already exists', 'danger')
        else:
            new_user = User(username=username, role=role)
            new_user.password = password
            db.session.add(new_user)
            db.session.commit()
            flash('User created successfully!', 'success')
            return redirect(url_for('auth.register'))

    # список ролей для селекта
    roles = [
        (ROLE_TECHNICIAN, "Technician"),
        (ROLE_USER, "User"),
        (ROLE_VIEWER, "Viewer"),
        (ROLE_ACCOUNTING, "Accounting"),
        (ROLE_ADMIN, "Admin"),
        (ROLE_SUPERADMIN, "Superadmin"),
    ]
    return render_template('register.html', roles=roles)


# WCCR_MY_TIME_V1 ROUTES
# WCCR_MY_TIME_REPORTS_V1
@auth_bp.route("/my-time", methods=["GET"])
@login_required
def my_time():
    from flask import abort, current_app
    from services.employee_time_policy import load_policy
    # WCCR_PAID_BREAK_POLICY_V3C
    paid_break_limit = load_policy()["paid_break_minutes"]
    from datetime import datetime, timedelta, timezone, time
    from zoneinfo import ZoneInfo
    from sqlalchemy import select, or_
    from itsdangerous import URLSafeTimedSerializer
    from services.employee_time_tracking_models import (
        EmployeeShift, EmployeeTimeEvent
    )

    if not EmployeeTimeTrackingService.eligible(current_user):
        abort(403)

    tz = ZoneInfo("America/Los_Angeles")
    now = datetime.now(timezone.utc)

    from services.employee_time_calculations import make_time_calculator
    (aware, la_time, seconds_between, summary, overlap, totals, utc_midnight) = make_time_calculator(paid_break_limit, tz)

    shift = EmployeeTimeTrackingService.get_open_shift(
        current_user.id
    )
    # WCCR_SMART_SHIFT_RECOVERY_4A
    recovery_candidate = (
        EmployeeTimeTrackingService.recovery_candidate(current_user)
        if shift is None else None
    )

    active_pause = None
    current = {
        "total": 0, "worked": 0, "lunch": 0, "break": 0,
        "actual_work": 0, "paid_time": 0,
        "paid_break": 0, "unpaid_break": 0
    }

    if shift:
        events = db.session.execute(
            select(EmployeeTimeEvent)
            .where(EmployeeTimeEvent.shift_id == shift.id)
            .order_by(EmployeeTimeEvent.id)
        ).scalars().all()

        current_info = summary(shift, events, now)
        current = totals(
            current_info, current_info["start"], now
        )
        active_pause = current_info["active_pause"]

    # WCCR_MY_TIME_FILTERS_V68
    from datetime import date
    from flask import flash

    today = now.astimezone(tz).date()
    period = request.args.get("period", "today").lower()

    allowed_periods = {
        "today", "yesterday", "week", "two_weeks",
        "month", "year", "custom"
    }

    # Preserve existing URLs.
    aliases = {"day": "today"}
    period = aliases.get(period, period)

    if period not in allowed_periods:
        period = "today"

    if period == "today":
        first_day = today
        last_day = today
    elif period == "yesterday":
        first_day = today - timedelta(days=1)
        last_day = first_day
    elif period == "week":
        first_day = today - timedelta(days=today.weekday())
        last_day = today
    elif period == "two_weeks":
        first_day = today - timedelta(days=13)
        last_day = today
    elif period == "month":
        first_day = today.replace(day=1)
        last_day = today
    elif period == "year":
        first_day = today.replace(month=1, day=1)
        last_day = today
    else:
        try:
            first_day = date.fromisoformat(
                request.args.get("from", "")
            )
            last_day = date.fromisoformat(
                request.args.get("to", "")
            )
            if first_day > last_day:
                raise ValueError("Start date is after end date")
            if last_day > today:
                raise ValueError("Future dates are not allowed")
            if (last_day - first_day).days > 3660:
                raise ValueError("Range exceeds 10 years")
        except (TypeError, ValueError):
            flash("Invalid date range. Showing today.", "warning")
            period = "today"
            first_day = today
            last_day = today

    next_day = last_day + timedelta(days=1)

    left = utc_midnight(first_day)
    right = min(now, utc_midnight(next_day))

    report_shifts = db.session.execute(
        select(EmployeeShift)
        .where(
            EmployeeShift.user_id == current_user.id,
            EmployeeShift.started_at < right,
            or_(
                EmployeeShift.ended_at.is_(None),
                EmployeeShift.ended_at > left,
            ),
        )
        .order_by(EmployeeShift.started_at)
    ).scalars().all()

    shift_ids = [s.id for s in report_shifts]
    events_by_shift = {sid: [] for sid in shift_ids}

    if shift_ids:
        all_events = db.session.execute(
            select(EmployeeTimeEvent)
            .where(EmployeeTimeEvent.shift_id.in_(shift_ids))
            .order_by(EmployeeTimeEvent.shift_id, EmployeeTimeEvent.id)
        ).scalars().all()

        for event in all_events:
            events_by_shift[event.shift_id].append(event)

    prepared = [
        (
            s,
            summary(s, events_by_shift.get(s.id, []), now)
        )
        for s in report_shifts
    ]

    rows = []
    # WCCR_MY_TIME_REPORT_V664
    report_total = {
        "total": 0, "worked": 0,
        "actual_work": 0, "paid_time": 0,
        "lunch": 0, "break": 0,
        "paid_break": 0, "unpaid_break": 0
    }

    day = first_day
    while day < next_day and day <= today:
        day_left = utc_midnight(day)
        day_right = min(
            now, utc_midnight(day + timedelta(days=1))
        )

        amounts = {
            "total": 0, "worked": 0,
            "actual_work": 0, "paid_time": 0,
            "lunch": 0, "break": 0,
            "paid_break": 0, "unpaid_break": 0
        }
        review = False
        open_shift = False
        idle = False

        for item, info in prepared:
            part = totals(info, day_left, day_right)

            if part["total"] <= 0:
                continue

            for key in amounts:
                amounts[key] += part[key]

            review |= bool(
                item.requires_review or info["anomalies"]
            )
            open_shift |= item.ended_at is None
            idle |= info["has_idle"]

        for key in report_total:
            report_total[key] += amounts[key]

        # WCCR_MY_TIME_REAL_DAYS_V1
        # Display only days with recorded time intervals.
        # Keep report totals and payroll calculations unchanged.
        if amounts["total"] > 0:
            rows.append({
                "date": day.strftime("%b %d, %Y"),
                **amounts,
                "review": review,
                "open": open_shift,
                "idle": idle,
            })

        day += timedelta(days=1)


    # Independent today summary, even for historical filters.
    today_left = utc_midnight(today)
    today_right = now

    if first_day <= today <= last_day:
        today_worked = next(
            (r["worked"] for r in rows
             if r["date"] == today.strftime("%b %d, %Y")),
            0
        )
    else:
        today_shifts = db.session.execute(
            select(EmployeeShift).where(
                EmployeeShift.user_id == current_user.id,
                EmployeeShift.started_at < today_right,
                or_(
                    EmployeeShift.ended_at.is_(None),
                    EmployeeShift.ended_at > today_left,
                ),
            )
        ).scalars().all()

        today_ids = [s.id for s in today_shifts]
        today_events = {sid: [] for sid in today_ids}

        if today_ids:
            events_today = db.session.execute(
                select(EmployeeTimeEvent)
                .where(EmployeeTimeEvent.shift_id.in_(today_ids))
                .order_by(
                    EmployeeTimeEvent.shift_id,
                    EmployeeTimeEvent.id
                )
            ).scalars().all()

            for event in events_today:
                today_events[event.shift_id].append(event)

        today_worked = 0
        for item in today_shifts:
            info = summary(
                item, today_events.get(item.id, []), now
            )
            today_worked += totals(
                info, today_left, today_right
            )["worked"]

    recent = db.session.execute(
        select(EmployeeShift)
        .where(EmployeeShift.user_id == current_user.id)
        .order_by(EmployeeShift.id.desc())
        .limit(5)
    ).scalars().all()

    serializer = URLSafeTimedSerializer(
        current_app.secret_key,
        salt="wccr-my-time-v1"
    )
    csrf = serializer.dumps({
        "uid": current_user.id,
        "sid": session.get("_id", "")
    })

    return render_template(
        "my_time.html",
        shift=shift,
        recent=recent,

        # WCCR_V73H4_UNIFIED_REVIEW
        attendance_review_statuses={
            row.shift_id: row.status
            for row in (
                db.session.execute(
                    __import__("sqlalchemy").select(
                        __import__(
                            "services.attendance_review_v73",
                            fromlist=["AttendanceReviewRevision"]
                        ).AttendanceReviewRevision
                    ).where(
                        __import__(
                            "services.attendance_review_v73",
                            fromlist=["AttendanceReviewRevision"]
                        ).AttendanceReviewRevision.shift_id.in_(
                            [item.id for item in recent]
                        )
                    ).order_by(
                        __import__(
                            "services.attendance_review_v73",
                            fromlist=["AttendanceReviewRevision"]
                        ).AttendanceReviewRevision.revision.desc(),
                        __import__(
                            "services.attendance_review_v73",
                            fromlist=["AttendanceReviewRevision"]
                        ).AttendanceReviewRevision.id.desc()
                    )
                ).scalars().all()
            )[::-1]
        },

        correction_history=db.session.execute(
            __import__("sqlalchemy").text(
                "SELECT id,shift_id,status,created_at_utc,review_note "
                "FROM employee_time_correction_requests "
                "WHERE user_id=:uid ORDER BY id DESC LIMIT 100"
            ),
            {"uid": current_user.id}
        ).mappings().all(),
        correction_can_edit=__import__(
            "services.attendance_review_permissions",
            fromlist=["can_review_action"]
        ).can_review_action(current_user, "edit"),
        correction_can_review=(
            __import__(
                "services.attendance_review_permissions",
                fromlist=["can_review_action"]
            ).can_review_action(current_user, "review")
            or __import__(
                "services.attendance_review_permissions",
                fromlist=["can_review_action"]
            ).can_review_action(current_user, "approve")
        ),
        correction_statuses={
            r[0]: r[1] for r in db.session.execute(
                __import__('sqlalchemy').text(
                    "SELECT shift_id,status FROM "
                    "employee_time_correction_requests "
                    "WHERE user_id=:uid AND status='pending'"
                ),
                {"uid": current_user.id}
            ).all()
        },
        started=la_time(shift.started_at) if shift else None,
        la_time=la_time,
        active_pause=active_pause,
        paid_break_limit_seconds=paid_break_limit * 60,
        paid_break_limit_minutes=paid_break_limit,
        worked_seconds=current["worked"],
        current=current,
        period=period,
        filter_from=first_day.isoformat(),
        filter_to=last_day.isoformat(),
        report_includes_today=first_day <= today <= last_day,
        today_date_label=today.strftime("%b %d, %Y"),
        report_rows=list(reversed(rows)),
        today_worked=today_worked,
        report_total=report_total,
        report_label=(
            first_day.strftime("%b %d, %Y")
            if first_day == last_day
            else first_day.strftime("%b %d, %Y")
            + " - "
            + last_day.strftime("%b %d, %Y")
        ),
        csrf_token=csrf,
        recovery_candidate=recovery_candidate,
        employee_clock_mode=current_user.employee_time_tracking_mode,
    )


@auth_bp.route("/my-time/action", methods=["POST"])
@login_required
def my_time_action():
    from flask import abort, current_app
    from itsdangerous import (
        URLSafeTimedSerializer, BadSignature, SignatureExpired
    )
    from services.employee_time_tracking_service import TimeTrackingError

    if not EmployeeTimeTrackingService.eligible(current_user):
        abort(403)

    serializer = URLSafeTimedSerializer(
        current_app.secret_key,
        salt="wccr-my-time-v1"
    )

    try:
        token_data = serializer.loads(
            request.form.get("employee_time_csrf", ""),
            max_age=1800
        )
    except (BadSignature, SignatureExpired):
        abort(400)

    if (
        not isinstance(token_data, dict)
        or token_data.get("uid") != current_user.id
        or token_data.get("sid") != session.get("_id", "")
    ):
        abort(400)

    action = request.form.get("action", "")
    allowed = {
        "clock_in", "clock_out", "lunch_start", "lunch_end",
        "break_start", "break_end"
    }
    if action not in allowed:
        abort(400)

    try:
        # WCCR_CLOCK_V66B
        if action == "clock_in":
            if EmployeeTimeTrackingService.get_open_shift(
                current_user.id
            ) is not None:
                raise TimeTrackingError("Already clocked in")
            EmployeeTimeTrackingService.login(current_user)

        elif action == "clock_out":
            EmployeeTimeTrackingService.end_shift(current_user)

        elif action == "lunch_start":
            EmployeeTimeTrackingService.start_pause(
                current_user, "LUNCH"
            )

        elif action == "lunch_end":
            EmployeeTimeTrackingService.end_pause(
                current_user, "LUNCH"
            )

        elif action == "break_start":
            EmployeeTimeTrackingService.start_pause(
                current_user, "BREAK"
            )

        elif action == "break_end":
            EmployeeTimeTrackingService.end_pause(
                current_user, "BREAK"
            )

    except TimeTrackingError as exc:
        db.session.rollback()
        flash(str(exc), "warning")

    except Exception:
        db.session.rollback()
        logging.exception(
            "My Time action failed for user_id=%s",
            current_user.id
        )
        flash("Time tracking update failed.", "danger")

    else:
        flash("Time tracking updated.", "success")

    return redirect(url_for("auth.my_time"))
# WCCR_MY_TIME_V1 ROUTES END



# WCCR_TIME_POLICY_SETTINGS_V3B
@auth_bp.route("/time-tracking/settings", methods=["GET", "POST"])
@login_required
def time_tracking_settings():
    from flask import abort, current_app
    from itsdangerous import (
        URLSafeTimedSerializer, BadSignature, SignatureExpired
    )
    from services.employee_time_policy import (
        load_policy, save_policy, TimePolicyError
    )

    # WCCR_TIME_TRACKING_ERP_3D
    from services.erp_access_service import ErpAccessService

    can_manage = ErpAccessService.is_allowed(
        current_user,
        "erp.time_tracking.manage_settings",
        default_allowed=False,
    )
    can_view = can_manage or ErpAccessService.is_allowed(
        current_user,
        "erp.time_tracking.view_settings",
        default_allowed=False,
    )

    if not can_view:
        abort(403)

    if request.method == "POST" and not can_manage:
        abort(403)

    serializer = URLSafeTimedSerializer(
        current_app.secret_key,
        salt="wccr-time-policy-settings-v1"
    )

    if request.method == "POST":
        try:
            token = request.form.get("policy_csrf", "")
            payload = serializer.loads(token, max_age=3600)

            if (
                payload.get("uid") != current_user.id
                or payload.get("sid") != session.get("_id", "")
            ):
                abort(403)

        except (BadSignature, SignatureExpired, AttributeError):
            abort(403)

        try:
            values = {
                "paid_break_minutes": int(
                    request.form.get("paid_break_minutes", "")
                ),
                "idle_minutes": int(
                    request.form.get("idle_minutes", "")
                ),
                "auto_close_time": request.form.get(
                    "auto_close_time", ""
                ),
                "missing_lunch_after_hours": int(
                    request.form.get("missing_lunch_after_hours", "")
                ),
                "expected_lunch_minutes": int(
                    request.form.get("expected_lunch_minutes", "")
                ),
                "smart_shift_recovery": (
                    request.form.get("smart_shift_recovery") == "on"
                ),
            }

            save_policy(values)

        except (ValueError, TypeError, TimePolicyError) as exc:
            flash("Settings not saved: " + str(exc), "danger")

        except OSError:
            current_app.logger.exception(
                "Time tracking settings save failed"
            )
            flash("Settings could not be saved.", "danger")

        else:
            flash("Time Tracking Settings saved.", "success")

        return redirect(url_for("auth.time_tracking_settings"))

    try:
        policy = load_policy()
    except TimePolicyError:
        current_app.logger.exception(
            "Time tracking settings load failed"
        )
        flash("Stored settings are invalid.", "danger")
        policy = None

    csrf_token = serializer.dumps({
        "uid": current_user.id,
        "sid": session.get("_id", "")
    })

    return render_template(
        "time_tracking_settings.html",
        policy=policy,
        csrf_token=csrf_token,
        can_manage=can_manage,
    )


__all__ = ['auth_bp']


# WCCR_MY_TIME_CORRECTION_V41
@auth_bp.route("/my-time/correction/<int:shift_id>",
               methods=["GET", "POST"])
@login_required
def my_time_correction(shift_id):
    from flask import abort, current_app
    from itsdangerous import (
        URLSafeTimedSerializer, BadSignature, SignatureExpired
    )
    from sqlalchemy import text
    from sqlalchemy.exc import IntegrityError
    from datetime import datetime, timezone, timedelta
    from zoneinfo import ZoneInfo
    from services.employee_time_tracking_models import EmployeeShift

    if not EmployeeTimeTrackingService.eligible(current_user):
        abort(403)

    shift = db.session.get(EmployeeShift, shift_id)
    if shift is None or shift.user_id != current_user.id:
        abort(404)
    if shift.ended_at is None:
        abort(409)

    la = ZoneInfo("America/Los_Angeles")

    def utc(value):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def local(value):
        return utc(value).astimezone(la).strftime(
            "%Y-%m-%dT%H:%M"
        )

    def parse_local(value):
        try:
            naive = datetime.strptime(value, "%Y-%m-%dT%H:%M")
        except (ValueError, TypeError):
            raise ValueError("Invalid date or time.")

        candidates = set()
        for fold in (0, 1):
            candidate = naive.replace(tzinfo=la, fold=fold)
            instant = candidate.astimezone(timezone.utc)
            if instant.astimezone(la).replace(
                tzinfo=None
            ) == naive:
                candidates.add(instant)

        if len(candidates) != 1:
            raise ValueError(
                "Ambiguous or nonexistent Pacific time."
            )
        return candidates.pop()

    serializer = URLSafeTimedSerializer(
        current_app.secret_key,
        salt="wccr-my-time-correction-v41"
    )

    old_start = utc(shift.started_at)
    old_end = utc(shift.ended_at)

    if request.method == "POST":
        required = {
            "csrf_token", "started_at", "ended_at",
            "reason", "expected_start", "expected_end"
        }
        if set(request.form.keys()) != required:
            abort(400)

        try:
            token = serializer.loads(
                request.form["csrf_token"], max_age=1800
            )
        except (BadSignature, SignatureExpired):
            abort(400)

        if not isinstance(token, dict) or (
            token.get("uid") != current_user.id
            or token.get("shift") != shift_id
            or token.get("sid") != session.get("_id", "")
        ):
            abort(400)

        try:
            reason = request.form["reason"].strip()
            start = parse_local(request.form["started_at"])
            end = parse_local(request.form["ended_at"])

            if not 4 <= len(reason) <= 1000:
                raise ValueError(
                    "Reason must contain 4 to 1000 characters."
                )
            if end <= start or end - start > timedelta(hours=24):
                raise ValueError(
                    "Invalid interval (maximum 24 hours)."
                )
            if end > datetime.now(timezone.utc):
                raise ValueError("End cannot be in the future.")
            if start == old_start and end == old_end:
                raise ValueError("No changes requested.")
            if (
                request.form["expected_start"] != old_start.isoformat()
                or request.form["expected_end"] != old_end.isoformat()
            ):
                raise ValueError("Shift changed. Refresh the page.")

            db.session.execute(text(
                "INSERT INTO employee_time_correction_requests "
                "(shift_id,user_id,original_start_utc,"
                "original_end_utc,requested_start_utc,"
                "requested_end_utc,reason,status,created_at_utc) "
                "VALUES (:shift,:uid,:os,:oe,:ns,:ne,"
                ":reason,'pending',:created)"
            ), {
                "shift": shift_id,
                "uid": current_user.id,
                "os": old_start.isoformat(),
                "oe": old_end.isoformat(),
                "ns": start.isoformat(),
                "ne": end.isoformat(),
                "reason": reason,
                "created": datetime.now(timezone.utc).isoformat()
            })
            db.session.commit()
        except ValueError as exc:
            db.session.rollback()
            flash(str(exc), "warning")
        except IntegrityError:
            db.session.rollback()
            flash("A request is already pending.", "warning")
        except Exception:
            db.session.rollback()
            current_app.logger.exception(
                "Correction request failed: %s", shift_id
            )
            flash("Unable to submit request.", "danger")
        else:
            flash("Correction request submitted.", "success")
            return redirect(url_for("auth.my_time"))

    token = serializer.dumps({
        "uid": current_user.id,
        "shift": shift_id,
        "sid": session.get("_id", "")
    })

    return render_template(
        "employee_time_correction_request.html",
        shift=shift,
        started_at=local(shift.started_at),
        ended_at=local(shift.ended_at),
        expected_start=old_start.isoformat(),
        expected_end=old_end.isoformat(),
        csrf_token=token
    )


# WCCR_ATTENDANCE_V50_CORE
@auth_bp.route("/attendance/correction-requests", methods=["GET"])
@login_required
def attendance_correction_requests():
    from flask import abort, current_app
    from sqlalchemy import text
    from services.attendance_review_permissions import can_review_action
    from services.employee_time_tracking_models import EmployeeShift
    from models import User

    if not (
        can_review_action(current_user, "review")
        or can_review_action(current_user, "approve")
    ):
        abort(403)

    records = db.session.execute(text("""
        SELECT id, shift_id, user_id, status, reason,
               original_start_utc, original_end_utc,
               requested_start_utc, requested_end_utc,
               created_at_utc, reviewed_by_user_id,
               reviewed_at_utc, review_note
        FROM employee_time_correction_requests
        ORDER BY CASE WHEN status='pending' THEN 0 ELSE 1 END,
                 id DESC
        LIMIT 500
    """)).mappings().all()


    # WCCR_ATTENDANCE_V51_PREPARE
    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo
    from itsdangerous import URLSafeTimedSerializer

    pacific = ZoneInfo("America/Los_Angeles")
    serializer = URLSafeTimedSerializer(
        current_app.secret_key,
        salt="wccr-correction-review-v51"
    )

    def display_pt(value):
        if not value:
            return "-"
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(pacific).strftime(
            "%b %d, %Y %I:%M %p"
        )

    prepared = []
    for item in records:
        r = dict(item)
        r["original_start_pt"] = display_pt(
            r["original_start_utc"]
        )
        r["original_end_pt"] = display_pt(
            r["original_end_utc"]
        )
        r["requested_start_pt"] = display_pt(
            r["requested_start_utc"]
        )
        r["requested_end_pt"] = display_pt(
            r["requested_end_utc"]
        )
        r["employee_name"] = "Employee #" + str(r["user_id"])
        r["review_csrf"] = serializer.dumps({
            "uid": current_user.id,
            "rid": r["id"],
            "sid": session.get("_id", "")
        })
        prepared.append(r)

    return render_template(
        "attendance_correction_queue.html",
        records=prepared,
        can_edit=can_review_action(current_user, "edit"),
        can_approve=can_review_action(current_user, "approve")
    )


# WCCR_ATTENDANCE_V51_ACTIONS
@auth_bp.route(
    "/attendance/correction-requests/<int:request_id>/reject",
    methods=["POST"]
)
@login_required
def attendance_correction_reject(request_id):
    from flask import abort, current_app
    from sqlalchemy import text
    from itsdangerous import (
        URLSafeTimedSerializer, BadSignature, SignatureExpired
    )
    from datetime import datetime, timezone
    from services.attendance_review_permissions import can_review_action

    if not can_review_action(current_user, "approve"):
        abort(403)

    serializer = URLSafeTimedSerializer(
        current_app.secret_key,
        salt="wccr-correction-review-v51"
    )
    try:
        payload = serializer.loads(
            request.form.get("csrf_token", ""), max_age=1800
        )
    except (BadSignature, SignatureExpired):
        abort(400)

    if (
        not isinstance(payload, dict)
        or payload.get("uid") != current_user.id
        or payload.get("rid") != request_id
        or payload.get("sid") != session.get("_id", "")
    ):
        abort(400)

    note = request.form.get("review_note", "").strip()
    if not 4 <= len(note) <= 1000:
        flash("Review reason must be 4-1000 characters.", "warning")
        return redirect(url_for("auth.attendance_correction_requests"))

    try:
        result = db.session.execute(text("""
            UPDATE employee_time_correction_requests
            SET status='rejected',
                reviewed_by_user_id=:uid,
                reviewed_at_utc=:now,
                review_note=:note
            WHERE id=:rid AND status='pending'
        """), {
            "uid": current_user.id,
            "now": datetime.now(timezone.utc).isoformat(),
            "note": note,
            "rid": request_id
        })
        if result.rowcount != 1:
            db.session.rollback()
            flash("Request already processed or not found.", "warning")
        else:
            db.session.commit()
            flash("Correction request rejected.", "success")
    except Exception:
        db.session.rollback()
        current_app.logger.exception(
            "Correction rejection failed: %s", request_id
        )
        flash("Unable to reject request.", "danger")

    return redirect(url_for("auth.attendance_correction_requests"))


# WCCR_ATTENDANCE_V52_REVIEW
@auth_bp.route(
    "/attendance/correction-requests/<int:request_id>",
    methods=["GET"]
)
@login_required
def attendance_correction_detail(request_id):
    from flask import abort
    from sqlalchemy import text, select
    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo
    from services.attendance_review_permissions import can_review_action
    from services.employee_time_tracking_models import (
        EmployeeShift, EmployeeTimeEvent
    )

    can_review = can_review_action(current_user, "review")
    can_edit = can_review_action(current_user, "edit")
    can_approve = can_review_action(current_user, "approve")

    if not (can_review or can_edit or can_approve):
        abort(403)

    request_record = db.session.execute(text("""
        SELECT *
        FROM employee_time_correction_requests
        WHERE id=:rid
    """), {"rid": request_id}).mappings().first()

    if request_record is None:
        abort(404)

    record = dict(request_record)
    shift = db.session.get(EmployeeShift, record["shift_id"])

    if shift is None or shift.user_id != record["user_id"]:
        abort(409)

    events = db.session.execute(
        select(EmployeeTimeEvent)
        .where(EmployeeTimeEvent.shift_id == shift.id)
        .order_by(EmployeeTimeEvent.id)
    ).scalars().all()

    pacific = ZoneInfo("America/Los_Angeles")

    def pt(value):
        if value is None:
            return "-"
        if isinstance(value, str):
            value = datetime.fromisoformat(value)
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(pacific).strftime(
            "%b %d, %Y %I:%M %p"
        )

    event_rows = [
        {
            "type": event.event_type,
            "time": pt(event.occurred_at),
            "note": event.note or ""
        }
        for event in events
    ]

    return render_template(
        "attendance_correction_detail.html",
        record=record,
        shift=shift,
        events=event_rows,
        pt=pt,
        can_edit=can_edit,
        can_approve=can_approve
    )


# WCCR_ATTENDANCE_V54_INTERVAL
@auth_bp.route(
    "/attendance/shift/<int:shift_id>/intervals",
    methods=["GET", "POST"]
)
@login_required
def attendance_interval_editor(shift_id):
    from flask import abort, current_app
    from sqlalchemy import text, select
    from datetime import datetime, timezone, timedelta
    from zoneinfo import ZoneInfo
    from itsdangerous import (
        URLSafeTimedSerializer, BadSignature, SignatureExpired
    )
    from services.attendance_review_permissions import can_review_action
    from services.employee_time_tracking_models import (
        EmployeeShift, EmployeeTimeEvent
    )

    if not can_review_action(current_user, "edit"):
        abort(403)

    shift = db.session.get(EmployeeShift, shift_id)
    if shift is None:
        abort(404)
    if shift.ended_at is None:
        abort(409)

    pacific = ZoneInfo("America/Los_Angeles")

    def aware(value):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def pt(value):
        if value is None:
            return "-"
        if isinstance(value, str):
            value = datetime.fromisoformat(value)
        return aware(value).astimezone(pacific).strftime(
            "%b %d, %Y %I:%M %p"
        )

    def parse_pt(value):
        naive = datetime.strptime(value, "%Y-%m-%dT%H:%M")
        candidates = set()
        for fold in (0, 1):
            local = naive.replace(tzinfo=pacific, fold=fold)
            utc = local.astimezone(timezone.utc)
            if utc.astimezone(pacific).replace(
                tzinfo=None
            ) == naive:
                candidates.add(utc)
        if len(candidates) != 1:
            raise ValueError(
                "Invalid or ambiguous Pacific Time."
            )
        return candidates.pop()

    serializer = URLSafeTimedSerializer(
        current_app.secret_key,
        salt="wccr-interval-v54"
    )

    start_bound = aware(shift.started_at)
    end_bound = aware(shift.ended_at)

    if request.method == "POST":
        if set(request.form.keys()) != {
            "csrf_token", "action_type", "start",
            "end", "reason"
        }:
            abort(400)

        try:
            token = serializer.loads(
                request.form["csrf_token"],
                max_age=1800
            )
        except (BadSignature, SignatureExpired):
            abort(400)

        if not isinstance(token, dict) or (
            token.get("uid") != current_user.id
            or token.get("shift") != shift_id
            or token.get("sid") != session.get("_id", "")
        ):
            abort(400)

        try:
            action = request.form["action_type"]
            reason = request.form["reason"].strip()
            start = parse_pt(request.form["start"])
            end = parse_pt(request.form["end"])

            if action not in (
                "correct_interval", "exclude_interval"
            ):
                raise ValueError("Invalid correction type.")
            if not 4 <= len(reason) <= 1000:
                raise ValueError(
                    "Reason must contain 4-1000 characters."
                )
            if not start_bound <= start < end <= end_bound:
                raise ValueError(
                    "Interval must be inside the closed shift."
                )
            if end - start > timedelta(hours=24):
                raise ValueError(
                    "Interval exceeds 24 hours. "
                    "Split it into smaller intervals."
                )

            db.session.execute(text("""
                INSERT INTO employee_time_interval_proposals
                (
                    shift_id, created_by_user_id,
                    action_type, start_utc, end_utc,
                    reason, status, created_at_utc
                )
                VALUES
                (
                    :shift, :uid, :action, :start,
                    :end, :reason, 'pending', :created
                )
            """), {
                "shift": shift_id,
                "uid": current_user.id,
                "action": action,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "reason": reason,
                "created": datetime.now(
                    timezone.utc
                ).isoformat()
            })
            db.session.commit()
            flash(
                "Interval proposal saved for review.",
                "success"
            )
            return redirect(url_for(
                "auth.attendance_interval_editor",
                shift_id=shift_id
            ))
        except ValueError as exc:
            db.session.rollback()
            flash(str(exc), "warning")
        except Exception:
            db.session.rollback()
            current_app.logger.exception(
                "Interval proposal failed: %s", shift_id
            )
            flash("Unable to save proposal.", "danger")

    events = db.session.execute(
        select(EmployeeTimeEvent)
        .where(EmployeeTimeEvent.shift_id == shift_id)
        .order_by(
            EmployeeTimeEvent.occurred_at,
            EmployeeTimeEvent.id
        )
    ).scalars().all()

    proposals = db.session.execute(text("""
        SELECT id, action_type, start_utc, end_utc,
               reason, status, created_at_utc,
               created_by_user_id
        FROM employee_time_interval_proposals
        WHERE shift_id=:shift
        ORDER BY id DESC
        LIMIT 200
    """), {"shift": shift_id}).mappings().all()

    token = serializer.dumps({
        "uid": current_user.id,
        "shift": shift_id,
        "sid": session.get("_id", "")
    })

    return render_template(
        "attendance_interval_editor.html",
        shift=shift,
        events=events,
        proposals=proposals,
        pt=pt,
        csrf_token=token,
        started_at=start_bound.astimezone(
            pacific
        ).strftime("%Y-%m-%dT%H:%M"),
        ended_at=end_bound.astimezone(
            pacific
        ).strftime("%Y-%m-%dT%H:%M")
    )


# WCCR_ATTENDANCE_V55_TARGETED
@auth_bp.route(
    "/attendance/shift/<int:shift_id>/targeted-correction",
    methods=["GET", "POST"]
)
@login_required
def attendance_targeted_correction(shift_id):
    from flask import abort, current_app
    from sqlalchemy import text, select
    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo
    from itsdangerous import (
        URLSafeTimedSerializer, BadSignature, SignatureExpired
    )
    from services.attendance_review_permissions import can_review_action
    from services.employee_time_tracking_models import (
        EmployeeShift, EmployeeTimeEvent
    )

    if not can_review_action(current_user, "edit"):
        abort(403)

    shift = db.session.get(EmployeeShift, shift_id)
    if shift is None:
        abort(404)
    if shift.ended_at is None:
        abort(409)

    pacific = ZoneInfo("America/Los_Angeles")

    def utc(value):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def display(value):
        if value is None:
            return "-"
        if isinstance(value, str):
            value = datetime.fromisoformat(value)
        return utc(value).astimezone(pacific).strftime(
            "%b %d, %Y %I:%M %p"
        )

    def local_input(value):
        return utc(value).astimezone(pacific).strftime(
            "%Y-%m-%dT%H:%M"
        )

    def parse_local(value):
        try:
            naive = datetime.strptime(value, "%Y-%m-%dT%H:%M")
        except (ValueError, TypeError):
            raise ValueError("Invalid date/time.")

        candidates = set()
        for fold in (0, 1):
            local = naive.replace(tzinfo=pacific, fold=fold)
            instant = local.astimezone(timezone.utc)
            if instant.astimezone(pacific).replace(
                tzinfo=None
            ) == naive:
                candidates.add(instant)

        if len(candidates) != 1:
            raise ValueError(
                "Ambiguous or nonexistent Pacific time."
            )
        return candidates.pop()

    serializer = URLSafeTimedSerializer(
        current_app.secret_key,
        salt="wccr-targeted-v55"
    )

    events = db.session.execute(
        select(EmployeeTimeEvent)
        .where(EmployeeTimeEvent.shift_id == shift_id)
        .order_by(
            EmployeeTimeEvent.occurred_at,
            EmployeeTimeEvent.id
        )
    ).scalars().all()

    # WCCR_V73H3C_ONE_CLICK
    if (
        request.method == "POST"
        and request.form.get("action_type") == "save_and_approve_review"
    ):
        import json as _one_click_json
        from flask import jsonify
        from services.attendance_review_v73 import AttendanceReviewError
        from services.attendance_review_save_v73 import save_revision
        from services.attendance_review_approve_v73 import approve_revision

        if not (
            can_review_action(current_user, "edit")
            and can_review_action(current_user, "approve")
        ):
            abort(403)

        if set(request.form.keys()) != {
            "csrf_token", "action_type", "payload"
        }:
            abort(400)

        try:
            _token = serializer.loads(
                request.form["csrf_token"], max_age=1800
            )
        except (BadSignature, SignatureExpired):
            abort(400)

        if not isinstance(_token, dict) or (
            _token.get("uid") != current_user.id
            or _token.get("shift") != shift_id
            or _token.get("sid") != session.get("_id", "")
        ):
            abort(400)

        try:
            raw = request.form["payload"]
            if len(raw) > 30000:
                raise AttendanceReviewError("Payload too large.")

            data = _one_click_json.loads(raw)
            if not isinstance(data, dict) or set(data) != {
                "expected_revision", "expected_source",
                "started_at", "ended_at", "periods", "reason"
            }:
                raise AttendanceReviewError(
                    "Invalid correction payload."
                )

            revision_id, number = save_revision(
                shift_id=shift_id,
                actor_id=current_user.id,
                expected_revision=data["expected_revision"],
                expected_source=data["expected_source"],
                started_at=data["started_at"],
                ended_at=data["ended_at"],
                periods=data["periods"],
                reason=data["reason"],
                commit=False,
            )

            result = approve_revision(
                shift_id=shift_id,
                actor_id=current_user.id,
                revision_id=revision_id,
                expected_source=data["expected_source"],
                commit=False,
            )

            db.session.commit()
            return jsonify({"ok": True, **result})

        except (
            AttendanceReviewError, ValueError, TypeError, KeyError
        ) as exc:
            db.session.rollback()
            return jsonify({
                "ok": False, "error": str(exc)
            }), 409
        except Exception:
            db.session.rollback()
            current_app.logger.exception(
                "V7.3H.3C approval failed: shift=%s", shift_id
            )
            return jsonify({
                "ok": False,
                "error": "Unable to approve shift."
            }), 500

    # WCCR_ATTENDANCE_V73F_APPROVE
    if (
        request.method == "POST"
        and request.form.get("action_type") == "approve_review_revision"
    ):
        from flask import jsonify
        from services.attendance_review_v73 import AttendanceReviewError
        from services.attendance_review_approve_v73 import approve_revision

        if not can_review_action(current_user, "approve"):
            abort(403)

        if set(request.form.keys()) != {
            "csrf_token", "action_type",
            "revision_id", "expected_source"
        }:
            abort(400)

        try:
            _approval_token = serializer.loads(
                request.form["csrf_token"], max_age=1800
            )
        except (BadSignature, SignatureExpired):
            abort(400)

        if not isinstance(_approval_token, dict) or (
            _approval_token.get("uid") != current_user.id
            or _approval_token.get("shift") != shift_id
            or _approval_token.get("sid") != session.get("_id", "")
        ):
            abort(400)

        try:
            _raw_id = request.form["revision_id"]
            if not _raw_id.isdecimal():
                raise AttendanceReviewError(
                    "Invalid revision ID."
                )

            _result = approve_revision(
                shift_id=shift_id,
                actor_id=current_user.id,
                revision_id=int(_raw_id),
                expected_source=request.form["expected_source"],
            )

            return jsonify({
                "ok": True,
                **_result
            })

        except AttendanceReviewError as exc:
            db.session.rollback()
            return jsonify({
                "ok": False,
                "error": str(exc)
            }), 409
        except Exception:
            db.session.rollback()
            current_app.logger.exception(
                "V7.3F approval failed: shift=%s", shift_id
            )
            return jsonify({
                "ok": False,
                "error": "Unable to approve revision."
            }), 500

    # WCCR_ATTENDANCE_V73C_SAVE
    if (
        request.method == "POST"
        and request.form.get("action_type") == "save_review_revision"
    ):
        import json as _review_json
        from flask import jsonify
        from services.attendance_review_v73 import AttendanceReviewError
        from services.attendance_review_save_v73 import save_revision

        if not can_review_action(current_user, "edit"):
            abort(403)

        required_save = {
            "csrf_token", "action_type", "payload"
        }
        if set(request.form.keys()) != required_save:
            abort(400)

        try:
            _token = serializer.loads(
                request.form["csrf_token"], max_age=1800
            )
        except (BadSignature, SignatureExpired):
            abort(400)

        if not isinstance(_token, dict) or (
            _token.get("uid") != current_user.id
            or _token.get("shift") != shift_id
            or _token.get("sid") != session.get("_id", "")
        ):
            abort(400)

        try:
            raw = request.form["payload"]
            if len(raw) > 30000:
                raise AttendanceReviewError("Payload too large.")

            data = _review_json.loads(raw)
            if not isinstance(data, dict) or set(data) != {
                "expected_revision", "expected_source",
                "started_at", "ended_at", "periods", "reason"
            }:
                raise AttendanceReviewError("Invalid correction payload.")

            revision_id, revision_number = save_revision(
                shift_id=shift_id,
                actor_id=current_user.id,
                expected_revision=data["expected_revision"],
                expected_source=data["expected_source"],
                started_at=data["started_at"],
                ended_at=data["ended_at"],
                periods=data["periods"],
                reason=data["reason"],
            )
            return jsonify({
                "ok": True,
                "revision_id": revision_id,
                "revision": revision_number,
                "status": "pending",
            })

        except (
            AttendanceReviewError, ValueError,
            TypeError, KeyError
        ) as exc:
            db.session.rollback()
            return jsonify({
                "ok": False,
                "error": str(exc),
            }), 409
        except Exception:
            db.session.rollback()
            current_app.logger.exception(
                "V7.3C save failed: shift=%s", shift_id
            )
            return jsonify({
                "ok": False,
                "error": "Unable to save correction."
            }), 500

    if request.method == "POST":
        required = {
            "csrf_token", "action_type", "event_id",
            "new_time", "reason"
        }
        if set(request.form.keys()) != required:
            abort(400)

        try:
            payload = serializer.loads(
                request.form["csrf_token"], max_age=1800
            )
        except (BadSignature, SignatureExpired):
            abort(400)

        if not isinstance(payload, dict) or (
            payload.get("uid") != current_user.id
            or payload.get("shift") != shift_id
            or payload.get("sid") != session.get("_id", "")
        ):
            abort(400)

        try:
            action = request.form["action_type"]
            reason = request.form["reason"].strip()

            if not 4 <= len(reason) <= 1000:
                raise ValueError(
                    "Reason must contain 4-1000 characters."
                )

            event_id = None
            original = None
            proposed = None

            if action == "correct_event_time":
                try:
                    event_id = int(request.form["event_id"])
                except (ValueError, TypeError):
                    raise ValueError("Select an event.")

                event = next(
                    (e for e in events if e.id == event_id),
                    None
                )
                if event is None:
                    raise ValueError("Event does not belong to shift.")

                original = utc(event.occurred_at)
                proposed = parse_local(
                    request.form["new_time"]
                )

                if proposed == original:
                    raise ValueError("No change requested.")

                if not (
                    utc(shift.started_at) <= proposed
                    <= utc(shift.ended_at)
                ):
                    raise ValueError(
                        "Proposed time must be within shift boundaries."
                    )

            elif action == "exclude_entire_shift":
                if (
                    request.form["event_id"]
                    or request.form["new_time"]
                ):
                    raise ValueError(
                        "Exclusion must not specify an event or time."
                    )
            else:
                raise ValueError("Invalid action.")

            db.session.execute(text("""
                INSERT INTO employee_time_targeted_proposals
                (
                    shift_id, event_id, action_type,
                    original_time_utc, proposed_time_utc,
                    created_by_user_id, reason,
                    status, created_at_utc
                )
                VALUES
                (
                    :shift, :event, :action,
                    :original, :proposed, :uid,
                    :reason, 'pending', :created
                )
            """), {
                "shift": shift_id,
                "event": event_id,
                "action": action,
                "original": (
                    original.isoformat()
                    if original else None
                ),
                "proposed": (
                    proposed.isoformat()
                    if proposed else None
                ),
                "uid": current_user.id,
                "reason": reason,
                "created": datetime.now(
                    timezone.utc
                ).isoformat()
            })
            db.session.commit()
            flash(
                "Targeted correction proposal saved.",
                "success"
            )
            return redirect(url_for(
                "auth.attendance_targeted_correction",
                shift_id=shift_id
            ))
        except ValueError as exc:
            db.session.rollback()
            flash(str(exc), "warning")
        except sqlite3.IntegrityError:
            db.session.rollback()
            flash("A pending proposal already exists.", "warning")
        except Exception:
            db.session.rollback()
            current_app.logger.exception(
                "Targeted correction failed: %s", shift_id
            )
            flash(
                "Unable to save proposal. "
                "Check for an existing pending proposal.",
                "danger"
            )

    proposals = db.session.execute(text("""
        SELECT id, event_id, action_type,
               original_time_utc, proposed_time_utc,
               reason, status, created_at_utc
        FROM employee_time_targeted_proposals
        WHERE shift_id=:shift
        ORDER BY id DESC LIMIT 200
    """), {"shift": shift_id}).mappings().all()

    token = serializer.dumps({
        "uid": current_user.id,
        "shift": shift_id,
        "sid": session.get("_id", "")
    })

    from services.attendance_timeline_preview import (
        reconstruct_preview
    )
    timeline = reconstruct_preview(events)

    # WCCR V7.3D: read-only source/version context.
    from services.attendance_review_save_v73 import (
        source_fingerprint, latest_revision_number
    )
    review_source = source_fingerprint(shift, events)
    review_revision = latest_revision_number(shift_id)


    # WCCR_ATTENDANCE_V73E_RESTORE
    # The latest saved revision is the editor's visible draft.
    # Source shift/events remain immutable.
    import json as _v73_json
    from types import SimpleNamespace as _V73NS
    from services.attendance_review_v73 import (
        AttendanceReviewRevision as _V73Revision,
        AttendanceReviewPeriod as _V73Period,
    )

    display_shift = shift
    display_timeline = timeline
    saved_review = None

    _latest_review = db.session.execute(
        select(_V73Revision)
        .where(_V73Revision.shift_id == shift_id)
        .order_by(
            _V73Revision.revision.desc(),
            _V73Revision.id.desc()
        )
        .limit(1)
    ).scalar_one_or_none()

    if _latest_review is not None:
        _saved_periods = db.session.execute(
            select(_V73Period)
            .where(
                _V73Period.revision_id == _latest_review.id
            )
            .order_by(
                _V73Period.started_at_utc,
                _V73Period.id
            )
        ).scalars().all()

        _rows = []

        for _period in _saved_periods:
            _ids = _v73_json.loads(
                _period.source_event_ids or "[]"
            )

            if not isinstance(_ids, list):
                raise ValueError(
                    "Invalid saved period source references."
                )

            _rows.append(_V73NS(
                type=_period.kind,
                start=local_input(_period.started_at_utc),
                end=local_input(_period.ended_at_utc),
                event_ids=_ids,
                saved_start=local_input(
                    _period.started_at_utc
                ),
                saved_end=local_input(
                    _period.ended_at_utc
                ),
            ))

        display_shift = _V73NS(
            started_at=_latest_review.started_at_utc,
            ended_at=_latest_review.ended_at_utc,
        )

        display_timeline = _V73NS(rows=_rows)

        saved_review = {
            "id": _latest_review.id,
            "revision": _latest_review.revision,
            "status": _latest_review.status,
            "reason": _latest_review.reason,
        }

    return render_template(
        "attendance_targeted_correction.html",
        shift=shift,
        events=events,
        timeline=timeline,
        proposals=proposals,
        display=display,
        local_input=local_input,
        csrf_token=token,
        review_source=review_source,
        review_revision=review_revision,
        display_shift=display_shift,
        display_timeline=display_timeline,
        saved_review=saved_review
    )
