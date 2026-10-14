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

__all__ = ['auth_bp']
