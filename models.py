from datetime import datetime, date
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy.orm import relationship, validates
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy import CheckConstraint, UniqueConstraint, func
from extensions import db

# --------------------------------
# User roles
# --------------------------------
ROLE_SUPERADMIN = 'superadmin'
ROLE_ADMIN      = 'admin'
ROLE_USER       = 'user'
ROLE_VIEWER     = 'viewer'
ROLE_TECHNICIAN = 'technician'

# Разрешённые значения ролей
ALLOWED_ROLES = {
    ROLE_SUPERADMIN,
    ROLE_ADMIN,
    ROLE_TECHNICIAN,
    ROLE_USER,
    ROLE_VIEWER,
}

# Синонимы/варианты написания
ROLE_ALIASES = {
    'tech': ROLE_TECHNICIAN,
    'technician': ROLE_TECHNICIAN,
    'super': ROLE_SUPERADMIN,
    'sa': ROLE_SUPERADMIN,
    'admin': ROLE_ADMIN,
    'administrator': ROLE_ADMIN,
    'viewer': ROLE_VIEWER,
    'read_only': ROLE_VIEWER,
    'user': ROLE_USER,
    'employee': ROLE_USER,
}


# --------------------------------
# Users
# --------------------------------
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    role = db.Column(db.String(20), nullable=False, default=ROLE_TECHNICIAN)

    @validates("role")
    def _validate_role(self, key, value: str | None):
        v = (value or "").strip().lower()
        v = ROLE_ALIASES.get(v, v)
        return v if v in ALLOWED_ROLES else ROLE_TECHNICIAN

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
# Work Orders
# --------------------------------
class WorkOrder(db.Model):
    __tablename__ = "work_orders"
    __table_args__ = {"extend_existing": True}

    id = db.Column(db.Integer, primary_key=True)
    technician_id = db.Column(db.Integer, db.ForeignKey("user.id"), index=True, nullable=True)
    technician = relationship("User", lazy="joined")

    units = db.relationship(
        "WorkUnit",
        backref="work_order",
        cascade="all, delete-orphan",
        lazy="selectin",
        overlaps="parts,work_order",
    )

    parts = db.relationship(
        "WorkOrderPart",
        backref="work_order",
        primaryjoin="WorkOrder.id==WorkOrderPart.work_order_id",
        cascade="all, delete-orphan",
        lazy="selectin",
        overlaps="unit,parts,work_order",
    )

    technician_name = db.Column(db.String(80), nullable=False)
    job_numbers     = db.Column(db.String(120), nullable=False)
    brand           = db.Column(db.String(40))
    model           = db.Column(db.String(25))
    serial          = db.Column(db.String(25))
    job_type        = db.Column(db.String(16), default="BASE")
    delivery_fee    = db.Column(db.Float, default=0.0)
    markup_percent  = db.Column(db.Float, default=0.0)
    status          = db.Column(db.String(20), default="search_ordered")
    created_at      = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at      = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    ordered_date    = db.Column(db.Date, nullable=True)
    customer_po = db.Column(db.String(64), nullable=True, index=True)

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

    @property
    def technician_username(self) -> str:
        if self.technician and self.technician.username:
            return self.technician.username
        return self.technician_name or ""

    def set_technician(self, user: "User"):
        self.technician = user
        self.technician_id = user.id
        self.technician_name = user.username


class WorkUnit(db.Model):
    __tablename__ = "work_units"
    __table_args__ = {"extend_existing": True}

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
        overlaps="parts,work_order",
    )


class WorkOrderPart(db.Model):
    __tablename__ = "work_order_parts"
    __table_args__ = {"extend_existing": True}

    id = db.Column(db.Integer, primary_key=True)
    work_order_id = db.Column(db.Integer, db.ForeignKey("work_orders.id"), nullable=False, index=True)
    unit_id = db.Column(db.Integer, db.ForeignKey("work_units.id"), nullable=True, index=True)
    part_number = db.Column(db.String(80), nullable=False)
    part_name   = db.Column(db.String(120))
    quantity    = db.Column(db.Integer, default=1)
    alt_part_numbers = db.Column(db.String(200))
    alt_numbers      = db.Column(db.String(200))
    supplier       = db.Column(db.String(80))
    backorder_flag = db.Column(db.Boolean, default=False)
    status      = db.Column(db.String(32), default="search_ordered")
    line_status = db.Column(db.String(32), default="search_ordered")
    unit_price_base  = db.Column(db.Float)
    unit_price_final = db.Column(db.Float)
    unit_cost        = db.Column(db.Float, nullable=True, default=None)
    issued_qty     = db.Column(db.Integer, nullable=False, default=0)
    last_issued_at = db.Column(db.DateTime, nullable=True)
    warehouse  = db.Column(db.String(120), nullable=True)
    unit_label = db.Column(db.String(120), nullable=True)
    stock_hint = db.Column(db.String(120))
    ordered_flag = db.Column(db.Boolean, default=False, index=True)
    ordered_date = db.Column(db.Date, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_insurance_supplied = db.Column(db.Boolean, nullable=False, default=False, index=True)
    invoice_number = db.Column(db.String(32), nullable=True, index=True)

    @property
    def warehouse_or_label(self) -> str:
        return self.warehouse or self.unit_label or ""

    @property
    def is_ordered(self) -> bool:
        st = (self.status or "").strip().lower()
        ls = (self.line_status or "").strip().lower()
        return bool(self.ordered_flag) or st == "ordered" or ls == "ordered"

    def mark_ordered(self, when: date | None = None):
        self.ordered_flag = True
        self.ordered_date = when or date.today()
        if (self.status or "").lower() != "ordered":
            self.status = "ordered"
        if (self.line_status or "").lower() != "ordered":
            self.line_status = "ordered"

    def clear_ordered(self):
        self.ordered_flag = False
        self.ordered_date = None

    def __repr__(self):
        return f"<WOP id={self.id} pn={self.part_number} ordered={self.ordered_flag} on={self.ordered_date}>"


# --------------------------------
# Tech receive log
# --------------------------------
class TechReceiveLog(db.Model):
    __tablename__ = "tech_receive_log"
    __table_args__ = {"extend_existing": True}

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
    __tablename__ = "part"
    __table_args__ = {"extend_existing": True}

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    part_number = db.Column(db.String(100), unique=True, nullable=False, index=True)
    quantity = db.Column(db.Integer, default=0)
    unit_cost = db.Column(db.Float, nullable=False)
    location = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Recipient(db.Model):
    __tablename__ = "recipient"
    __table_args__ = {"extend_existing": True}

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)


# --------------------------------
# Issued parts
# --------------------------------
class IssuedPartRecord(db.Model):
    __tablename__ = 'issued_part_record'
    __table_args__ = {"extend_existing": True}

    id = db.Column(db.Integer, primary_key=True)
    part_id = db.Column(db.Integer, db.ForeignKey('part.id'), nullable=False, index=True)
    quantity = db.Column(db.Integer, nullable=False)
    issued_to = db.Column(db.String(255), nullable=False, index=True)
    issued_by = db.Column(db.String(255), nullable=False)
    reference_job = db.Column(db.String(255), index=True)
    issue_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, index=True)
    unit_cost_at_issue = db.Column(db.Float, nullable=False)

    # техподтверждение (как было)
    confirmed_by_tech = db.Column(db.Boolean, default=False, nullable=False)
    confirmed_at = db.Column(db.DateTime)
    confirmed_by = db.Column(db.String(64))

    invoice_number = db.Column(db.Integer, index=True)
    location = db.Column(db.String(120), index=True)

    # связь с батчем
    batch_id = db.Column(db.Integer, db.ForeignKey('issued_batch.id', ondelete='SET NULL'), index=True)
    batch = db.relationship("IssuedBatch", back_populates="parts", lazy="joined")

    part  = db.relationship('Part', backref=db.backref('issued_records', lazy=True))

    is_insurance_supplied = db.Column(db.Boolean, nullable=False, default=False, index=True)

    # --- НОВЫЕ ПОЛЯ (соответствуют миграции) ---
    consumed_qty  = db.Column(db.Integer, nullable=True)             # 0..quantity
    consumed_flag = db.Column(db.Boolean, nullable=False, default=False)
    consumed_at   = db.Column(db.DateTime, nullable=True)
    consumed_by   = db.Column(db.String(120), nullable=True)
    consumed_note = db.Column(db.String(500), nullable=True)
    inv_ref = db.Column(db.String(32), nullable=True, index=True)  # INV# из WorkOrderPart

    # На какой job № было помечено списание (для отчёта Grouped)
    consumed_job_ref = db.Column(db.String(64), nullable=True, index=True)
    # журнальные записи по частичным списаниям (по job)
    consumption_logs = db.relationship(
        "IssuedConsumptionLog",
        back_populates="issued_part",
        lazy="select",
        cascade="all, delete-orphan",
    )

    # --- Удобные свойства/методы ---

    @property
    def remaining_qty(self) -> int:
        q = int(self.quantity or 0)
        used = int(self.consumed_qty or 0)
        return max(0, q - used)

    @property
    def status_tuple(self) -> tuple[str, int, int]:
        """('OPEN'|'PARTIAL'|'CONSUMED', consumed_qty, quantity)"""
        q = int(self.quantity or 0)
        used = int(self.consumed_qty or 0)
        if used <= 0:
            return ("OPEN", 0, q)
        if used >= q and q > 0:
            return ("CONSUMED", q, q)
        return ("PARTIAL", used, q)

    @property
    def status_label(self) -> str:
        kind, used, q = self.status_tuple
        if kind == "OPEN":
            return "OPEN"
        if kind == "CONSUMED":
            return f"CONSUMED {q}/{q}"
        return f"PARTIAL {used}/{q}"

    def _sync_flag(self):
        q = int(self.quantity or 0)
        used = int(self.consumed_qty or 0)
        self.consumed_flag = (q > 0 and used >= q)

    def apply_consume(self, delta: int, user: str | None = None, note: str | None = None) -> bool:
        """Увеличить списание на delta (>=1), не превышая quantity. Возвращает: были ли изменения."""
        d = int(delta or 0)
        if d <= 0:
            return False
        q = int(self.quantity or 0)
        used = int(self.consumed_qty or 0)
        new_used = min(q, used + d)
        if new_used == used:
            return False
        self.consumed_qty = new_used
        self.consumed_at = datetime.utcnow()
        if user:
            self.consumed_by = (user or "").strip()[:120]
        if note:
            self.consumed_note = (note or "").strip()[:500]
        self._sync_flag()
        return True

    def unconsume_all(self) -> bool:
        """Сбросить списание по строке (для отката)."""
        changed = bool(
            self.consumed_qty
            or self.consumed_flag
            or self.consumed_at
            or self.consumed_by
            or self.consumed_note
            or (self.consumption_logs and len(self.consumption_logs) > 0)
        )
        self.consumed_qty = None
        self.consumed_flag = False
        self.consumed_at = None
        self.consumed_by = None
        self.consumed_note = None

        # сами объекты логов будут удалены через cascade="all, delete-orphan"
        self.consumption_logs.clear()
        return changed



    def __repr__(self):
        return f"<IssuedPartRecord id={self.id} inv={self.invoice_number} batch={self.batch_id}>"

class IssuedConsumptionLog(db.Model):
    __tablename__ = "issued_consumption_log"
    __table_args__ = {"extend_existing": True}

    id = db.Column(db.Integer, primary_key=True)

    # какая строка инвойса (IssuedPartRecord) была частично списана
    issued_part_id = db.Column(
        db.Integer,
        db.ForeignKey("issued_part_record.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # сколько штук списали этой операцией
    qty = db.Column(db.Integer, nullable=False)

    # на какой job-номер списали (например "987568", "9855" и т.д.)
    job_ref = db.Column(db.String(64), nullable=True, index=True)

    # мета-инфо
    consumed_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, index=True)
    consumed_by = db.Column(db.String(120), nullable=True)
    note = db.Column(db.String(500), nullable=True)

    # связь обратно к строке инвойса
    issued_part = db.relationship(
        "IssuedPartRecord",
        back_populates="consumption_logs",
        lazy="joined",
    )

    def __repr__(self):
        return f"<IssuedConsumptionLog id={self.id} part_id={self.issued_part_id} qty={self.qty} job={self.job_ref}>"

# --------------------------------
# External Order tracking
# --------------------------------
class OrderItem(db.Model):
    __tablename__ = "order_items"
    __table_args__ = {"extend_existing": True}

    id = db.Column(db.Integer, primary_key=True)
    order_number = db.Column(db.String(64), index=True, nullable=False)
    technician   = db.Column(db.String(128), index=True, nullable=False)
    supplier     = db.Column(db.String(128))
    part_number  = db.Column(db.String(128), index=True, nullable=False)
    part_name    = db.Column(db.String(256))
    qty_ordered  = db.Column(db.Integer, default=1, nullable=False)
    unit_cost    = db.Column(db.Float)
    location     = db.Column(db.String(100), index=True)
    status       = db.Column(db.String(32), index=True, default="ordered")
    date_ordered = db.Column(db.DateTime, default=datetime.utcnow)
    date_received= db.Column(db.DateTime)
    notes        = db.Column(db.Text)
    row_key      = db.Column(db.String(512), unique=True)

# ---- вставь это место сразу после IssuedPartRecord (или рядом с ним) ----

class IssuedBatch(db.Model):
    __tablename__ = "issued_batch"
    __table_args__ = {'extend_existing': True}

    id = db.Column(db.Integer, primary_key=True)
    invoice_number = db.Column(db.Integer, unique=True, index=True, nullable=False)
    issued_to = db.Column(db.String(255), nullable=False)
    issued_by = db.Column(db.String(255), nullable=False)
    reference_job = db.Column(db.String(255))
    issue_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    location = db.Column(db.String(120))

    # связи: одна «шапка» на много строк выдачи
    parts = db.relationship("IssuedPartRecord", back_populates="batch", lazy="selectin")

    # --- поля учёта списаний (из миграции 11072025_add_stock) ---
    is_stock      = db.Column(db.Boolean, nullable=False, default=False)
    consumed_flag = db.Column(db.Boolean, nullable=False, default=False)
    consumed_at   = db.Column(db.DateTime, nullable=True)
    consumed_by   = db.Column(db.String(120), nullable=True)
    consumed_note = db.Column(db.String(500), nullable=True)

    # --- удобные свойства/методы для UI ---

    @property
    def is_stock_inferred(self) -> bool:
        """Подстраховка: если is_stock не проставлен в старых данных — считаем по reference_job."""
        ref = (self.reference_job or "").strip().lower()
        return bool(self.is_stock or ref.startswith("stock"))

    @property
    def aggregate_consumption(self) -> dict:
        """Сумма по строкам: всего, списано, остаток."""
        q_total = sum(int(r.quantity or 0) for r in (self.parts or []))
        used_total = sum(int(r.consumed_qty or 0) for r in (self.parts or []))
        rem_total = max(0, q_total - used_total)
        return {"qty_total": q_total, "used_total": used_total, "remaining_total": rem_total}

    @property
    def status_label(self) -> str:
        """OPEN / PARTIAL x/y / CONSUMED для всего инвойса."""
        agg = self.aggregate_consumption
        q, used = agg["qty_total"], agg["used_total"]
        if q <= 0 or used <= 0:
            return "OPEN"
        if used >= q:
            return "CONSUMED"
        return f"PARTIAL {used}/{q}"

    def _sync_flag(self):
        agg = self.aggregate_consumption
        self.consumed_flag = (agg["qty_total"] > 0 and agg["used_total"] >= agg["qty_total"])

    def mark_consumed_meta(self, user: str | None = None, note: str | None = None):
        """Обновить метаданные списания (когда, кем, примечание) и синхронизировать флаг."""
        self.consumed_at = datetime.utcnow()
        if user:
            self.consumed_by = (user or "").strip()[:120]
        if note:
            self.consumed_note = (note or "").strip()[:500]
        self._sync_flag()

    def __repr__(self):
        return f"<IssuedBatch id={self.id} invoice={self.invoice_number} to={self.issued_to}>"

# --------------------------------
# Goods Receipts (приход)
# --------------------------------
# --- GoodsReceipt (без изменений полей) ---
class GoodsReceipt(db.Model):
    __tablename__ = "goods_receipts"
    __table_args__ = (
        db.Index("ix_gr_supplier_invoice", "supplier_name", "invoice_number"),
        {"extend_existing": True},
    )

    id = db.Column(db.Integer, primary_key=True)
    supplier_name = db.Column(db.String(200), nullable=False, index=True)
    invoice_number = db.Column(db.String(64))
    invoice_date = db.Column(db.Date)
    currency = db.Column(db.String(8), default="USD")
    notes = db.Column(db.Text)
    status = db.Column(db.String(16), default="draft", nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    created_by = db.Column(db.Integer)
    posted_at = db.Column(db.DateTime)
    posted_by = db.Column(db.Integer)
    attachment_path = db.Column(db.String(512))
    extra_expenses = db.Column(db.Float, default=0.0)

    # удобный алиас: .items == .lines
    @property
    def items(self):
        return self.lines

    @hybrid_property
    def total_cost(self) -> float:
        try:
            lines = getattr(self, "lines", None) or []
            return float(sum(
                (getattr(it, "quantity", 0) or 0) * (getattr(it, "unit_cost", 0.0) or 0.0)
                for it in lines
            ))
        except Exception:
            return 0.0


class GoodsReceiptLine(db.Model):
    __tablename__ = "goods_receipt_lines"
    __table_args__ = {"extend_existing": True}

    id = db.Column(db.Integer, primary_key=True)
    goods_receipt_id = db.Column(
        db.Integer,
        db.ForeignKey("goods_receipts.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    line_no    = db.Column(db.Integer, default=1)
    part_number= db.Column(db.String(120), nullable=False, index=True)
    part_name  = db.Column(db.String(255))
    quantity   = db.Column(db.Integer, default=1, nullable=False)
    unit_cost  = db.Column(db.Float, default=0.0)
    location   = db.Column(db.String(64))

    applied_qty = db.Column(db.Integer, default=0, nullable=False)

    goods_receipt = db.relationship(
        "GoodsReceipt",
        backref=db.backref("lines", cascade="all, delete-orphan")
    )

    @hybrid_property
    def line_total(self) -> float:
        try:
            return float((self.quantity or 0) * (self.unit_cost or 0.0))
        except Exception:
            return 0.0

# --------------------------------
# Supplier Returns (новые таблицы)
# --------------------------------
class SupplierReturnBatch(db.Model):
    __tablename__ = "supplier_return_batch"
    __table_args__ = {"extend_existing": True}

    id = db.Column(db.Integer, primary_key=True)

    # кто/что возвращаем
    supplier_name = db.Column(db.String(200), index=True)
    reference_receiving_id = db.Column(db.Integer)  # опционально, без FK, просто справочно

    # статусы и мета
    status = db.Column(db.String(20), nullable=False, default="draft")  # draft|posted
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    created_by = db.Column(db.String(120))
    posted_at = db.Column(db.DateTime)
    posted_by = db.Column(db.String(120))
    tech_note = db.Column(db.String(255))  # NEW: technician + job note

    # агрегаты
    total_items = db.Column(db.Integer, nullable=False, default=0)
    total_value = db.Column(db.Float,   nullable=False, default=0.0)

    # строки возврата
    items = db.relationship(
        "SupplierReturnItem",
        back_populates="batch",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def __repr__(self):
        return f"<SupplierReturnBatch id={self.id} status={self.status} supplier={self.supplier_name!r}>"


class SupplierReturnItem(db.Model):
    __tablename__ = "supplier_return_item"
    __table_args__ = {"extend_existing": True}

    id = db.Column(db.Integer, primary_key=True)

    batch_id = db.Column(
        db.Integer,
        db.ForeignKey("supplier_return_batch.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    part_number = db.Column(db.String(120), nullable=False, index=True)
    part_name   = db.Column(db.String(255))
    location    = db.Column(db.String(120), index=True)

    qty_returned = db.Column(db.Integer, nullable=False, default=0)
    unit_cost    = db.Column(db.Float,   nullable=False, default=0.0)
    total_cost   = db.Column(db.Float,   nullable=False, default=0.0)
    tech_note = db.Column(db.String(255))  # Tech / Job per line

    batch = db.relationship("SupplierReturnBatch", back_populates="items", lazy="joined")

    def __repr__(self):
        return f"<SupplierReturnItem id={self.id} pn={self.part_number!r} qty={self.qty_returned} loc={self.location!r}>"

# --- Backwards-compatible aliases (как у тебя) ---
ReceivingBatch = GoodsReceipt
ReceivingItem  = GoodsReceiptLine



