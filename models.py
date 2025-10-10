from datetime import datetime, time
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

from extensions import db
from sqlalchemy import Column, Integer, String, Boolean, Float, DateTime, ForeignKey


# --------------------------------
# User roles
# --------------------------------
ROLE_SUPERADMIN = 'superadmin'
ROLE_ADMIN = 'admin'
ROLE_USER = 'user'
ROLE_VIEWER = 'viewer'


# --------------------------------
# Users
# --------------------------------
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    role = db.Column(db.String(20), nullable=False, default=ROLE_USER)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    @property
    def password(self):
        raise AttributeError('password is not a readable attribute')

    @password.setter
    def password(self, password):
        self.set_password(password)


# --------------------------------
# Work Orders (шапка)
# --------------------------------
class WorkOrder(db.Model):
    __tablename__ = "work_orders"
    id = db.Column(db.Integer, primary_key=True)

    # multi-appliance: список юнитов (по приборам)
    units = db.relationship(
        "WorkUnit",
        backref="work_order",
        cascade="all, delete-orphan",
        lazy="selectin",
        overlaps="parts,work_order"
    )

    # «плоские» строки (старый/простой режим) — из той же таблицы work_order_parts
    parts = db.relationship(
        "WorkOrderPart",
        backref="work_order",
        primaryjoin="WorkOrder.id==WorkOrderPart.work_order_id",
        cascade="all, delete-orphan",
        lazy="selectin",
        overlaps="unit,parts,work_order"
    )

    technician_name = db.Column(db.String(80), nullable=False)
    job_numbers = db.Column(db.String(120), nullable=False)  # "98256, 98356"
    brand = db.Column(db.String(40))
    model = db.Column(db.String(25))
    serial = db.Column(db.String(25))

    job_type = db.Column(db.String(16), default="BASE")  # BASE | INSURANCE
    delivery_fee = db.Column(db.Float, default=0.0)
    markup_percent = db.Column(db.Float, default=0.0)
    status = db.Column(db.String(20), default="search_ordered")  # search_ordered | ordered | done

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    @property
    def canonical_job(self) -> str:
        nums = [n.strip() for n in (self.job_numbers or "").split(",") if n.strip()]
        ints = []
        for n in nums:
            try:
                ints.append(int(n))
            except ValueError:
                pass
        return str(max(ints)) if ints else (nums[0] if nums else "")


# --------------------------------
# Work Units (несколько аппаратов в одном WO)
# --------------------------------
class WorkUnit(db.Model):
    __tablename__ = "work_units"
    id = db.Column(db.Integer, primary_key=True)
    work_order_id = db.Column(db.Integer, db.ForeignKey("work_orders.id"), nullable=False, index=True)

    brand  = db.Column(db.String(80))
    model  = db.Column(db.String(25))
    serial = db.Column(db.String(25))

    parts = db.relationship(
        "WorkOrderPart",
        backref="unit",
        primaryjoin="WorkUnit.id==WorkOrderPart.unit_id",
        cascade="all, delete-orphan",
        lazy="selectin",
        overlaps="parts,work_order"
    )


# --------------------------------
# Work Order Part (единственный класс строк)
# --------------------------------
class WorkOrderPart(db.Model):
    __tablename__ = "work_order_parts"

    id = db.Column(db.Integer, primary_key=True)

    work_order_id = db.Column(db.Integer, db.ForeignKey("work_orders.id"), nullable=False, index=True)
    unit_id       = db.Column(db.Integer, db.ForeignKey("work_units.id"),  nullable=True,  index=True)

    # Данные по запчасти
    part_number = db.Column(db.String(80),  nullable=False)
    part_name   = db.Column(db.String(120))
    quantity    = db.Column(db.Integer, default=1)

    # ОРИГИНАЛЬНОЕ имя колонки (оставляем как есть)
    alt_part_numbers = db.Column(db.String(200))

    supplier       = db.Column(db.String(80))
    backorder_flag = db.Column(db.Boolean, default=False)

    # ОРИГИНАЛЬНОЕ имя колонки (оставляем как есть)
    status = db.Column(db.String(32), default="search_ordered")

    # Доп. реквизиты
    unit_label       = db.Column(db.String(120), nullable=True)  # историческое поле
    unit_price_base  = db.Column(db.Float)
    unit_price_final = db.Column(db.Float)
    unit_cost        = db.Column(db.Float, nullable=True, default=None)

    issued_qty = db.Column(db.Integer, nullable=False, default=0)  # сколько уже выдано по этой строке WO
    last_issued_at = db.Column(db.DateTime, nullable=True)

    # Новый реквизит склада
    warehouse       = db.Column(db.String(120), nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # ---- Совместимые алиасы для старого кода/UI ----
    @property
    def alt_numbers(self) -> str:
        return self.alt_part_numbers

    @alt_numbers.setter
    def alt_numbers(self, v: str):
        self.alt_part_numbers = v

    @property
    def line_status(self) -> str:
        return self.status

    @line_status.setter
    def line_status(self, v: str):
        self.status = v

    @property
    def warehouse_or_label(self) -> str:
        return self.warehouse or self.unit_label or ""


# --------------------------------
# Tech receive log
# --------------------------------
class TechReceiveLog(db.Model):
    __tablename__ = "tech_receive_log"
    id = db.Column(db.Integer, primary_key=True)
    work_order_id = db.Column(db.Integer, db.ForeignKey("work_orders.id"), nullable=False, index=True)
    work_order_part_id = db.Column(db.Integer, db.ForeignKey("work_order_parts.id"), nullable=False, index=True)
    qty_received = db.Column(db.Integer, default=0)
    received_by  = db.Column(db.String(80))
    received_at  = db.Column(db.DateTime, default=datetime.utcnow)


# --------------------------------
# Inventory Part
# --------------------------------
class Part(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    part_number = db.Column(db.String(100), unique=True, nullable=False, index=True)
    quantity = db.Column(db.Integer, default=0)
    unit_cost = db.Column(db.Float, nullable=False)
    location = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


# --------------------------------
# Recipient (tech/client)
# --------------------------------
class Recipient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)


# =========================
# IssuedPartRecord (строка инвойса)
# =========================
class IssuedPartRecord(db.Model):
    __tablename__ = 'issued_part_record'

    id                 = db.Column(db.Integer, primary_key=True)
    part_id            = db.Column(db.Integer, db.ForeignKey('part.id'), nullable=False, index=True)
    quantity           = db.Column(db.Integer, nullable=False)
    issued_to          = db.Column(db.String(255), nullable=False, index=True)
    issued_by          = db.Column(db.String(255), nullable=False)
    reference_job      = db.Column(db.String(255), index=True)
    issue_date         = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, index=True)
    unit_cost_at_issue = db.Column(db.Float, nullable=False)
    # models.py (IssuedPartRecord)
    confirmed_by_tech = db.Column(db.Boolean, default=False, nullable=False)
    confirmed_at = db.Column(db.DateTime, nullable=True)
    confirmed_by = db.Column(db.String(64), nullable=True)

    # новые поля для инвойсов / отчётов
    invoice_number     = db.Column(db.Integer, index=True, nullable=True)
    location           = db.Column(db.String(120), index=True)

    # batch (NULL для legacy)
    batch_id = db.Column(
        db.Integer,
        db.ForeignKey('issued_batch.id', ondelete='SET NULL'),
        index=True,
        nullable=True
    )
    batch = db.relationship(
        "IssuedBatch",
        back_populates="parts",
        lazy="joined"
    )

    part = db.relationship('Part', backref=db.backref('issued_records', lazy=True))

    def __repr__(self):
        return f"<IssuedPartRecord id={self.id} inv={self.invoice_number} batch={self.batch_id}>"


# =========================
# IssuedBatch (шапка инвойса)
# =========================
class IssuedBatch(db.Model):
    __tablename__ = "issued_batch"
    __table_args__ = {'extend_existing': True}

    id             = db.Column(db.Integer, primary_key=True)
    invoice_number = db.Column(db.Integer, unique=True, index=True, nullable=False)
    issued_to      = db.Column(db.String(255), nullable=False)
    issued_by      = db.Column(db.String(255), nullable=False)
    reference_job  = db.Column(db.String(255))
    issue_date     = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    location       = db.Column(db.String(120), nullable=True)

    # связь только через back_populates (никаких backref)
    parts = db.relationship(
        "IssuedPartRecord",
        back_populates="batch",
        lazy="selectin"
    )

    def __repr__(self):
        return f"<IssuedBatch id={self.id} invoice={self.invoice_number} to={self.issued_to}>"


# --------------------------------
# OrderItem (трекинг заказов снаружи)
# --------------------------------
class OrderItem(db.Model):
    __tablename__ = "order_items"
    id = db.Column(db.Integer, primary_key=True)

    order_number = db.Column(db.String(64), index=True, nullable=False)
    technician   = db.Column(db.String(128), index=True, nullable=False)
    supplier     = db.Column(db.String(128))
    part_number  = db.Column(db.String(128), index=True, nullable=False)
    part_name    = db.Column(db.String(256))
    qty_ordered  = db.Column(db.Integer, nullable=False, default=1)
    unit_cost    = db.Column(db.Float)
    location     = db.Column(db.String(100), index=True)

    status       = db.Column(db.String(32), index=True, default="ordered")
    date_ordered = db.Column(db.DateTime, default=datetime.utcnow)
    date_received= db.Column(db.DateTime)
    notes        = db.Column(db.Text)
    row_key      = db.Column(db.String(512), unique=True)















