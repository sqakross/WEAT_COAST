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

            # WCCR_EMPLOYEE_TIME_LOGIN_START
            if EmployeeTimeTrackingService.eligible(user):
                try:
                    EmployeeTimeTrackingService.login(user)
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
    if EmployeeTimeTrackingService.eligible(current_user):
        try:
            EmployeeTimeTrackingService.logout(current_user)
        except Exception:
            db.session.rollback()
            logging.exception(
                'Employee Time Tracking logout failed: user_id=%s',
                current_user.id,
            )
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

    def summary(shift, events, end_time):
        start = aware(shift.started_at)
        finish = aware(shift.ended_at) or end_time
        if finish < start:
            finish = start

        intervals = {"LUNCH": [], "BREAK": []}
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

        return {
            "start": start,
            "end": finish,
            "intervals": intervals,
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
        allowance = 15 * 60

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

        worked = max(0, total - lunch - unpaid_break)

        return {
            "total": int(total),
            "lunch": int(lunch),
            "break": int(break_total),
            "paid_break": int(paid_break),
            "unpaid_break": int(unpaid_break),
            "worked": int(worked),
        }

    def utc_midnight(day):
        return datetime.combine(
            day, time.min, tzinfo=tz
        ).astimezone(timezone.utc)

    shift = EmployeeTimeTrackingService.get_open_shift(
        current_user.id
    )

    active_pause = None
    current = {
        "total": 0, "worked": 0, "lunch": 0, "break": 0
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

    period = request.args.get("period", "week").lower()
    if period not in ("day", "week", "month"):
        period = "week"

    today = now.astimezone(tz).date()

    if period == "day":
        first_day = today
    elif period == "week":
        first_day = today - timedelta(days=today.weekday())
    else:
        first_day = today.replace(day=1)

    if period == "month":
        if first_day.month == 12:
            next_day = first_day.replace(
                year=first_day.year + 1, month=1
            )
        else:
            next_day = first_day.replace(
                month=first_day.month + 1
            )
    elif period == "week":
        next_day = first_day + timedelta(days=7)
    else:
        next_day = first_day + timedelta(days=1)

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
    report_total = {
        "total": 0, "worked": 0,
        "lunch": 0, "break": 0
    }

    day = first_day
    while day < next_day and day <= today:
        day_left = utc_midnight(day)
        day_right = min(
            now, utc_midnight(day + timedelta(days=1))
        )

        amounts = {
            "total": 0, "worked": 0,
            "lunch": 0, "break": 0
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

        rows.append({
            "date": day.strftime("%b %d, %Y"),
            **amounts,
            "review": review,
            "open": open_shift,
            "idle": idle,
        })

        day += timedelta(days=1)

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
        started=la_time(shift.started_at) if shift else None,
        la_time=la_time,
        active_pause=active_pause,
        worked_seconds=current["worked"],
        current=current,
        period=period,
        report_rows=list(reversed(rows)),
        report_total=report_total,
        report_label=(
            first_day.strftime("%b %d, %Y")
            if period == "day"
            else first_day.strftime("%b %d, %Y")
            + " - "
            + (next_day - timedelta(days=1)).strftime(
                "%b %d, %Y"
            )
        ),
        csrf_token=csrf,
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
        "start", "lunch_start", "lunch_end",
        "break_start", "break_end", "end_shift"
    }
    if action not in allowed:
        abort(400)

    try:
        if action == "start":
            if EmployeeTimeTrackingService.get_open_shift(
                current_user.id
            ) is not None:
                raise TimeTrackingError("Shift already active")
            EmployeeTimeTrackingService.login(current_user)

        elif action == "end_shift":
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


__all__ = ['auth_bp']
