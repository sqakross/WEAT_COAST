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

class WorkOrder(db.Model):
    __tablename__ = "work_orders"
    id = db.Column(db.Integer, primary_key=True)

    units = db.relationship(
        "WorkUnit",
        backref="order",
        cascade="all, delete-orphan",
        lazy="joined",
    )

    technician_name = db.Column(db.String(80), nullable=False)
    # один или два номера работ, через запятую (напр. "98256, 98356")
    job_numbers = db.Column(db.String(120), nullable=False)

    brand = db.Column(db.String(40))
    model = db.Column(db.String(25))
    serial = db.Column(db.String(25))

    # BASE | INSURANCE
    job_type = db.Column(db.String(16), default="BASE")
    # по умолчанию можно 0; админ может задать вручную
    delivery_fee = db.Column(db.Float, default=0.0)
    markup_percent = db.Column(db.Float, default=0.0)

    # общестадийный статус работы
    status = db.Column(db.String(20), default="search_ordered")  # search_ordered | ordered | done

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    parts = db.relationship("WorkOrderPart", backref="work_order", cascade="all, delete-orphan")

    @property
    def canonical_job(self) -> str:
        """
        Наше правило: если job_numbers содержит два значения,
        primary = большее по числовому сравнению.
        """
        nums = [n.strip() for n in (self.job_numbers or "").split(",") if n.strip()]
        ints = []
        for n in nums:
            try:
                ints.append(int(n))
            except ValueError:
                pass
        return str(max(ints)) if ints else (nums[0] if nums else "")


class WorkOrderPart(db.Model):
    __tablename__ = "work_order_parts"
    id = db.Column(db.Integer, primary_key=True)
    work_order_id = db.Column(db.Integer, db.ForeignKey("work_orders.id"), nullable=False)

    part_number = db.Column(db.String(50), nullable=False)       # основной PN
    alt_part_numbers = db.Column(db.String(200))                 # альтернативы через запятую (опционально)
    part_name = db.Column(db.String(120))
    quantity = db.Column(db.Integer, default=1)

    alt_part_numbers = db.Column(db.String(200))
    supplier = db.Column(db.String(80))                          # если не в стоке — откуда заказывать
    backorder_flag = db.Column(db.Boolean, default=False)        # отметка backorder
    status = db.Column(db.String(20), default="search_ordered")  # при желании — пер-строчный статус

    # опционально фиксируем цены (без/с наценкой и доставкой)
    unit_label = db.Column(db.String(120), nullable=True)
    unit_price_base = db.Column(db.Float)
    unit_price_final = db.Column(db.Float)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class TechReceiveLog(db.Model):
    __tablename__ = "tech_receive_log"
    id = db.Column(db.Integer, primary_key=True)
    work_order_id = db.Column(db.Integer, db.ForeignKey("work_orders.id"), nullable=False)
    work_order_part_id = db.Column(db.Integer, db.ForeignKey("work_order_parts.id"), nullable=False)

    qty_received = db.Column(db.Integer, default=0)
    received_by = db.Column(db.String(80))          # логин/имя техника
    received_at = db.Column(db.DateTime, default=datetime.utcnow)

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

# --- Order Items (для трекера заказов) ---
class OrderItem(db.Model):
    __tablename__ = "order_items"
    id = db.Column(db.Integer, primary_key=True)

    order_number = db.Column(db.String(64), index=True, nullable=False)   # Order #
    technician   = db.Column(db.String(128), index=True, nullable=False)  # Tech name
    supplier     = db.Column(db.String(128))
    part_number  = db.Column(db.String(128), index=True, nullable=False)
    part_name    = db.Column(db.String(256))
    qty_ordered  = db.Column(db.Integer, nullable=False, default=1)
    unit_cost    = db.Column(db.Float)
    location     = db.Column(db.String(100), index=True)

    status       = db.Column(db.String(32), index=True, default="ordered")  # ordered|received|partial
    date_ordered = db.Column(db.DateTime, default=datetime.utcnow)
    date_received= db.Column(db.DateTime)
    notes        = db.Column(db.Text)

    # Для идемпотентного синка из Excel (если подключишь позже)
    row_key      = db.Column(db.String(512), unique=True)

# --- NEW: multi-appliance per WorkOrder ---

class WorkUnit(db.Model):
    __tablename__ = "work_units"

    id = db.Column(db.Integer, primary_key=True)
    work_order_id = db.Column(db.Integer, db.ForeignKey("work_orders.id"), nullable=False)

    brand  = db.Column(db.String(80))
    model  = db.Column(db.String(25))
    serial = db.Column(db.String(25))

    parts = db.relationship(
        "WorkUnitPart",
        backref="unit",
        cascade="all, delete-orphan",
        lazy="joined",
    )


class WorkUnitPart(db.Model):
    __tablename__ = "work_unit_parts"

    id = db.Column(db.Integer, primary_key=True)
    work_unit_id = db.Column(db.Integer, db.ForeignKey("work_units.id"), nullable=False)

    part_number      = db.Column(db.String(50), nullable=False)
    part_name        = db.Column(db.String(120))
    quantity         = db.Column(db.Integer, default=1)
    alt_numbers      = db.Column(db.String(200))
    supplier         = db.Column(db.String(80))
    backorder_flag   = db.Column(db.Boolean, default=False)
    line_status      = db.Column(db.String(32), default="search_ordered")  # search_ordered|ordered|done






