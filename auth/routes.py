from flask import Blueprint, render_template, redirect, url_for, flash, request
from werkzeug.security import check_password_hash, generate_password_hash
from flask_login import login_user, logout_user, login_required, current_user
from extensions import db, login_manager
from models import User, ROLE_SUPERADMIN, ROLE_ADMIN, ROLE_USER, ROLE_VIEWER
from sqlalchemy import func

auth_bp = Blueprint('auth', __name__)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Login route
@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Сделать username insensitive to case
        user = User.query.filter(func.lower(User.username) == username.lower()).first()

        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('inventory.dashboard'))
        else:
            flash('Invalid username or password', 'danger')
    return render_template('login.html')


# Logout route
@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('auth.login'))

# Register new user (only by superadmin)
@auth_bp.route('/register', methods=['GET', 'POST'])
@login_required
def register():
    if current_user.role != ROLE_SUPERADMIN:
        flash('Access denied', 'danger')
        return redirect(url_for('inventory.dashboard'))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        role = request.form['role']
        if User.query.filter_by(username=username).first():
            flash('User already exists', 'danger')
        else:
            new_user = User(username=username, role=role)
            new_user.password = password  # Здесь вызывается сеттер, который хэширует пароль
            db.session.add(new_user)
            db.session.commit()
            flash('User created successfully!', 'success')
            return redirect(url_for('auth.register'))

    return render_template('register.html')
# auth/routes.py
__all__ = ['auth_bp']

