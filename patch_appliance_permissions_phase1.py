from pathlib import Path

path = Path("models.py")
text = path.read_text(encoding="utf-8")

# ============================================================
# 1. Add MANAGER role
# ============================================================

old_roles = """ROLE_SUPERADMIN = 'superadmin'
ROLE_ADMIN      = 'admin'
ROLE_USER       = 'user'
ROLE_VIEWER     = 'viewer'
ROLE_TECHNICIAN = 'technician'
ROLE_ACCOUNTING = "accounting"
"""

new_roles = """ROLE_SUPERADMIN = 'superadmin'
ROLE_MANAGER    = 'manager'
ROLE_ADMIN      = 'admin'
ROLE_USER       = 'user'
ROLE_VIEWER     = 'viewer'
ROLE_TECHNICIAN = 'technician'
ROLE_ACCOUNTING = "accounting"
"""

if old_roles not in text:
    raise SystemExit("ERROR: role block not found")

text = text.replace(old_roles, new_roles, 1)


old_allowed = """ALLOWED_ROLES = {
    ROLE_SUPERADMIN,
    ROLE_ADMIN,
    ROLE_TECHNICIAN,
    ROLE_USER,
    ROLE_VIEWER,
    ROLE_ACCOUNTING,
}
"""

new_allowed = """ALLOWED_ROLES = {
    ROLE_SUPERADMIN,
    ROLE_MANAGER,
    ROLE_ADMIN,
    ROLE_TECHNICIAN,
    ROLE_USER,
    ROLE_VIEWER,
    ROLE_ACCOUNTING,
}
"""

if old_allowed not in text:
    raise SystemExit("ERROR: ALLOWED_ROLES block not found")

text = text.replace(old_allowed, new_allowed, 1)


old_alias = """    'admin': ROLE_ADMIN,
    'administrator': ROLE_ADMIN,
"""

new_alias = """    'manager': ROLE_MANAGER,
    'mgr': ROLE_MANAGER,
    'admin': ROLE_ADMIN,
    'administrator': ROLE_ADMIN,
"""

if old_alias not in text:
    raise SystemExit("ERROR: ROLE_ALIASES insertion point not found")

text = text.replace(old_alias, new_alias, 1)


# ============================================================
# 2. Add Warehouse + Permission foundation
# ============================================================

anchor = """    @password.setter
    def password(self, password):
        self.set_password(password)

class JobReservation(db.Model):
"""

replacement = """    @password.setter
    def password(self, password):
        self.set_password(password)


# ============================================================
# Access Control Foundation
# Warehouse scope + granular permissions
# ============================================================

class Warehouse(db.Model):
    __tablename__ = "warehouse"
    __table_args__ = (
        db.UniqueConstraint("code", name="uq_warehouse_code"),
        db.UniqueConstraint("name", name="uq_warehouse_name"),
        {"extend_existing": True},
    )

    id = db.Column(db.Integer, primary_key=True)

    code = db.Column(
        db.String(40),
        nullable=False,
        unique=True,
        index=True,
    )

    name = db.Column(
        db.String(120),
        nullable=False,
        unique=True,
        index=True,
    )

    description = db.Column(
        db.String(255),
        nullable=True,
    )

    is_active = db.Column(
        db.Boolean,
        nullable=False,
        default=True,
        index=True,
    )

    created_at = db.Column(
        db.DateTime,
        nullable=False,
        default=datetime.utcnow,
    )

    created_by_id = db.Column(
        db.Integer,
        db.ForeignKey("user.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    updated_at = db.Column(
        db.DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    updated_by_id = db.Column(
        db.Integer,
        db.ForeignKey("user.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    created_by = db.relationship(
        "User",
        foreign_keys=[created_by_id],
        lazy="joined",
    )

    updated_by = db.relationship(
        "User",
        foreign_keys=[updated_by_id],
        lazy="joined",
    )

    @property
    def created_at_local(self):
        return utc_to_local(self.created_at)

    @property
    def updated_at_local(self):
        return utc_to_local(self.updated_at)

    def __repr__(self):
        return (
            f"<Warehouse id={self.id} "
            f"code={self.code!r} "
            f"name={self.name!r} "
            f"active={self.is_active}>"
        )


class UserWarehouseAccess(db.Model):
    __tablename__ = "user_warehouse_access"
    __table_args__ = (
        db.UniqueConstraint(
            "user_id",
            "warehouse_id",
            name="uq_user_warehouse_access_user_warehouse",
        ),
        db.Index(
            "ix_user_warehouse_access_user_active",
            "user_id",
            "is_active",
        ),
        {"extend_existing": True},
    )

    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("user.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    warehouse_id = db.Column(
        db.Integer,
        db.ForeignKey("warehouse.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    is_default = db.Column(
        db.Boolean,
        nullable=False,
        default=False,
    )

    is_active = db.Column(
        db.Boolean,
        nullable=False,
        default=True,
        index=True,
    )

    created_at = db.Column(
        db.DateTime,
        nullable=False,
        default=datetime.utcnow,
    )

    created_by_id = db.Column(
        db.Integer,
        db.ForeignKey("user.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    user = db.relationship(
        "User",
        foreign_keys=[user_id],
        lazy="joined",
        backref=db.backref(
            "warehouse_accesses",
            lazy="selectin",
            cascade="all, delete-orphan",
        ),
    )

    warehouse = db.relationship(
        "Warehouse",
        foreign_keys=[warehouse_id],
        lazy="joined",
        backref=db.backref(
            "user_accesses",
            lazy="selectin",
        ),
    )

    created_by = db.relationship(
        "User",
        foreign_keys=[created_by_id],
        lazy="joined",
    )

    @property
    def created_at_local(self):
        return utc_to_local(self.created_at)

    def __repr__(self):
        return (
            f"<UserWarehouseAccess "
            f"user_id={self.user_id} "
            f"warehouse_id={self.warehouse_id} "
            f"default={self.is_default} "
            f"active={self.is_active}>"
        )


class Permission(db.Model):
    __tablename__ = "permission"
    __table_args__ = (
        db.UniqueConstraint("code", name="uq_permission_code"),
        db.Index(
            "ix_permission_group_active",
            "group_name",
            "is_active",
        ),
        {"extend_existing": True},
    )

    id = db.Column(db.Integer, primary_key=True)

    code = db.Column(
        db.String(120),
        nullable=False,
        unique=True,
        index=True,
    )

    name = db.Column(
        db.String(120),
        nullable=False,
    )

    description = db.Column(
        db.String(500),
        nullable=True,
    )

    group_name = db.Column(
        db.String(80),
        nullable=False,
        default="general",
        index=True,
    )

    is_active = db.Column(
        db.Boolean,
        nullable=False,
        default=True,
        index=True,
    )

    sort_order = db.Column(
        db.Integer,
        nullable=False,
        default=100,
    )

    created_at = db.Column(
        db.DateTime,
        nullable=False,
        default=datetime.utcnow,
    )

    def __repr__(self):
        return (
            f"<Permission id={self.id} "
            f"code={self.code!r} "
            f"group={self.group_name!r}>"
        )


class UserPermission(db.Model):
    __tablename__ = "user_permission"
    __table_args__ = (
        db.UniqueConstraint(
            "user_id",
            "permission_id",
            name="uq_user_permission_user_permission",
        ),
        db.Index(
            "ix_user_permission_user",
            "user_id",
        ),
        {"extend_existing": True},
    )

    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("user.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    permission_id = db.Column(
        db.Integer,
        db.ForeignKey("permission.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    granted_by_id = db.Column(
        db.Integer,
        db.ForeignKey("user.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    granted_at = db.Column(
        db.DateTime,
        nullable=False,
        default=datetime.utcnow,
    )

    user = db.relationship(
        "User",
        foreign_keys=[user_id],
        lazy="joined",
        backref=db.backref(
            "permission_grants",
            lazy="selectin",
            cascade="all, delete-orphan",
        ),
    )

    permission = db.relationship(
        "Permission",
        foreign_keys=[permission_id],
        lazy="joined",
        backref=db.backref(
            "user_grants",
            lazy="selectin",
        ),
    )

    granted_by = db.relationship(
        "User",
        foreign_keys=[granted_by_id],
        lazy="joined",
    )

    @property
    def granted_at_local(self):
        return utc_to_local(self.granted_at)

    def __repr__(self):
        return (
            f"<UserPermission "
            f"user_id={self.user_id} "
            f"permission_id={self.permission_id}>"
        )


class UserPermissionAudit(db.Model):
    __tablename__ = "user_permission_audit"
    __table_args__ = (
        db.Index(
            "ix_user_permission_audit_target_date",
            "target_user_id",
            "created_at",
        ),
        db.Index(
            "ix_user_permission_audit_actor_date",
            "actor_user_id",
            "created_at",
        ),
        {"extend_existing": True},
    )

    id = db.Column(db.Integer, primary_key=True)

    target_user_id = db.Column(
        db.Integer,
        db.ForeignKey("user.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    action = db.Column(
        db.String(60),
        nullable=False,
        index=True,
    )

    permission_code = db.Column(
        db.String(120),
        nullable=True,
        index=True,
    )

    warehouse_id = db.Column(
        db.Integer,
        db.ForeignKey("warehouse.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    old_value = db.Column(
        db.String(255),
        nullable=True,
    )

    new_value = db.Column(
        db.String(255),
        nullable=True,
    )

    actor_user_id = db.Column(
        db.Integer,
        db.ForeignKey("user.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    actor_username = db.Column(
        db.String(64),
        nullable=True,
        index=True,
    )

    created_at = db.Column(
        db.DateTime,
        nullable=False,
        default=datetime.utcnow,
        index=True,
    )

    target_user = db.relationship(
        "User",
        foreign_keys=[target_user_id],
        lazy="joined",
    )

    actor_user = db.relationship(
        "User",
        foreign_keys=[actor_user_id],
        lazy="joined",
    )

    warehouse = db.relationship(
        "Warehouse",
        foreign_keys=[warehouse_id],
        lazy="joined",
    )

    @property
    def created_at_local(self):
        return utc_to_local(self.created_at)

    def __repr__(self):
        return (
            f"<UserPermissionAudit id={self.id} "
            f"target={self.target_user_id} "
            f"action={self.action!r} "
            f"by={self.actor_username!r}>"
        )


class JobReservation(db.Model):
"""

if anchor not in text:
    raise SystemExit("ERROR: User/JobReservation insertion point not found")

text = text.replace(anchor, replacement, 1)

path.write_text(text, encoding="utf-8")

print("OK: models.py patched")
print("Added role: manager")
print("Added models:")
print("  Warehouse")
print("  UserWarehouseAccess")
print("  Permission")
print("  UserPermission")
print("  UserPermissionAudit")
