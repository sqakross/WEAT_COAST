from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_user, logout_user, login_required, current_user
from sqlalchemy import func
from extensions import db, login_manager
from models import (
    User,
    ROLE_SUPERADMIN, ROLE_ADMIN, ROLE_USER, ROLE_VIEWER, ROLE_TECHNICIAN,
    ALLOWED_ROLES, ROLE_ALIASES
)

auth_bp = Blueprint('auth', __name__)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Login
@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter(func.lower(User.username) == username.lower()).first()
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('inventory.dashboard'))
        flash('Invalid username or password', 'danger')
    return render_template('login.html')

# Logout
@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('auth.login'))

# Register new user (superadmin only)
@auth_bp.route('/register', methods=['GET', 'POST'])
@login_required
def register():
    if current_user.role != ROLE_SUPERADMIN:
        flash('Access denied', 'danger')
        return redirect(url_for('inventory.dashboard'))

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
        (ROLE_TECHNICIAN, 'Technician'),
        (ROLE_USER, 'User'),
        (ROLE_VIEWER, 'Viewer'),
        (ROLE_ADMIN, 'Admin'),
        (ROLE_SUPERADMIN, 'Superadmin'),
    ]
    return render_template('register.html', roles=roles)

__all__ = ['auth_bp']
