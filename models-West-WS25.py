from extensions import db
from flask_login import UserMixin
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

# User roles
ROLE_SUPERADMIN = 'superadmin'
ROLE_ADMIN = 'admin'
ROLE_USER = 'user'
ROLE_VIEWER = 'viewer'

# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    role = db.Column(db.String(20), nullable=False, default=ROLE_USER)

    # Запись пароля хэшированным
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    # Проверка пароля
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    # Чтобы поле password не было напрямую доступно
    @property
    def password(self):
        raise AttributeError('password is not a readable attribute')

    @password.setter
    def password(self, password):
        self.set_password(password)

# Part (inventory) model
class Part(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    part_number = db.Column(db.String(100), unique=True, nullable=False)
    quantity = db.Column(db.Integer, default=0)
    unit_cost = db.Column(db.Float, nullable=False)
    location = db.Column(db.String(100))  # shelf/location
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Client or technician model
class Recipient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)

# Transactions model (issue record)
class IssuedPartRecord(db.Model):
    __tablename__ = 'issued_part_record'

    id = db.Column(db.Integer, primary_key=True)
    part_id = db.Column(db.Integer, db.ForeignKey('part.id'), nullable=False)
    quantity = db.Column(db.Integer, nullable=False)
    issued_to = db.Column(db.String(255), nullable=False)
    issued_by = db.Column(db.String(255), nullable=False)
    reference_job = db.Column(db.String(255))
    issue_date = db.Column(db.DateTime, nullable=False)

    # Добавляем новое поле для цены на момент выдачи
    unit_cost_at_issue = db.Column(db.Float, nullable=False)

    part = db.relationship('Part', backref=db.backref('issued_records', lazy=True))



