"""add warehouse and permission foundation

Revision ID: 28245b15c982
Revises: 6c4e2f8a1d90
Create Date: 2026-08-18 08:05:39.798812
"""

from alembic import op
import sqlalchemy as sa


revision = "28245b15c982"
down_revision = "6c4e2f8a1d90"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "permission",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("code", sa.String(length=120), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("description", sa.String(length=500), nullable=True),
        sa.Column("group_name", sa.String(length=80), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("sort_order", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("code", name="uq_permission_code"),
    )

    op.create_index(
        "ix_permission_code",
        "permission",
        ["code"],
        unique=True,
    )
    op.create_index(
        "ix_permission_group_active",
        "permission",
        ["group_name", "is_active"],
        unique=False,
    )
    op.create_index(
        "ix_permission_group_name",
        "permission",
        ["group_name"],
        unique=False,
    )
    op.create_index(
        "ix_permission_is_active",
        "permission",
        ["is_active"],
        unique=False,
    )

    op.create_table(
        "warehouse",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("code", sa.String(length=40), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("description", sa.String(length=255), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("created_by_id", sa.Integer(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("updated_by_id", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(
            ["created_by_id"],
            ["user.id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["updated_by_id"],
            ["user.id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("code", name="uq_warehouse_code"),
        sa.UniqueConstraint("name", name="uq_warehouse_name"),
    )

    op.create_index(
        "ix_warehouse_code",
        "warehouse",
        ["code"],
        unique=True,
    )
    op.create_index(
        "ix_warehouse_created_by_id",
        "warehouse",
        ["created_by_id"],
        unique=False,
    )
    op.create_index(
        "ix_warehouse_is_active",
        "warehouse",
        ["is_active"],
        unique=False,
    )
    op.create_index(
        "ix_warehouse_name",
        "warehouse",
        ["name"],
        unique=True,
    )
    op.create_index(
        "ix_warehouse_updated_by_id",
        "warehouse",
        ["updated_by_id"],
        unique=False,
    )

    op.create_table(
        "user_permission",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("permission_id", sa.Integer(), nullable=False),
        sa.Column("granted_by_id", sa.Integer(), nullable=True),
        sa.Column("granted_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["granted_by_id"],
            ["user.id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["permission_id"],
            ["permission.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["user.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "user_id",
            "permission_id",
            name="uq_user_permission_user_permission",
        ),
    )

    op.create_index(
        "ix_user_permission_granted_by_id",
        "user_permission",
        ["granted_by_id"],
        unique=False,
    )
    op.create_index(
        "ix_user_permission_permission_id",
        "user_permission",
        ["permission_id"],
        unique=False,
    )
    op.create_index(
        "ix_user_permission_user",
        "user_permission",
        ["user_id"],
        unique=False,
    )
    op.create_index(
        "ix_user_permission_user_id",
        "user_permission",
        ["user_id"],
        unique=False,
    )

    op.create_table(
        "user_permission_audit",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("target_user_id", sa.Integer(), nullable=True),
        sa.Column("action", sa.String(length=60), nullable=False),
        sa.Column("permission_code", sa.String(length=120), nullable=True),
        sa.Column("warehouse_id", sa.Integer(), nullable=True),
        sa.Column("old_value", sa.String(length=255), nullable=True),
        sa.Column("new_value", sa.String(length=255), nullable=True),
        sa.Column("actor_user_id", sa.Integer(), nullable=True),
        sa.Column("actor_username", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["actor_user_id"],
            ["user.id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["target_user_id"],
            ["user.id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["warehouse_id"],
            ["warehouse.id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index(
        "ix_user_permission_audit_action",
        "user_permission_audit",
        ["action"],
        unique=False,
    )
    op.create_index(
        "ix_user_permission_audit_actor_date",
        "user_permission_audit",
        ["actor_user_id", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_user_permission_audit_actor_user_id",
        "user_permission_audit",
        ["actor_user_id"],
        unique=False,
    )
    op.create_index(
        "ix_user_permission_audit_actor_username",
        "user_permission_audit",
        ["actor_username"],
        unique=False,
    )
    op.create_index(
        "ix_user_permission_audit_created_at",
        "user_permission_audit",
        ["created_at"],
        unique=False,
    )
    op.create_index(
        "ix_user_permission_audit_permission_code",
        "user_permission_audit",
        ["permission_code"],
        unique=False,
    )
    op.create_index(
        "ix_user_permission_audit_target_date",
        "user_permission_audit",
        ["target_user_id", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_user_permission_audit_target_user_id",
        "user_permission_audit",
        ["target_user_id"],
        unique=False,
    )
    op.create_index(
        "ix_user_permission_audit_warehouse_id",
        "user_permission_audit",
        ["warehouse_id"],
        unique=False,
    )

    op.create_table(
        "user_warehouse_access",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("warehouse_id", sa.Integer(), nullable=False),
        sa.Column("is_default", sa.Boolean(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("created_by_id", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(
            ["created_by_id"],
            ["user.id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["user.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["warehouse_id"],
            ["warehouse.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "user_id",
            "warehouse_id",
            name="uq_user_warehouse_access_user_warehouse",
        ),
    )

    op.create_index(
        "ix_user_warehouse_access_created_by_id",
        "user_warehouse_access",
        ["created_by_id"],
        unique=False,
    )
    op.create_index(
        "ix_user_warehouse_access_is_active",
        "user_warehouse_access",
        ["is_active"],
        unique=False,
    )
    op.create_index(
        "ix_user_warehouse_access_user_active",
        "user_warehouse_access",
        ["user_id", "is_active"],
        unique=False,
    )
    op.create_index(
        "ix_user_warehouse_access_user_id",
        "user_warehouse_access",
        ["user_id"],
        unique=False,
    )
    op.create_index(
        "ix_user_warehouse_access_warehouse_id",
        "user_warehouse_access",
        ["warehouse_id"],
        unique=False,
    )


def downgrade():
    op.drop_index(
        "ix_user_warehouse_access_warehouse_id",
        table_name="user_warehouse_access",
    )
    op.drop_index(
        "ix_user_warehouse_access_user_id",
        table_name="user_warehouse_access",
    )
    op.drop_index(
        "ix_user_warehouse_access_user_active",
        table_name="user_warehouse_access",
    )
    op.drop_index(
        "ix_user_warehouse_access_is_active",
        table_name="user_warehouse_access",
    )
    op.drop_index(
        "ix_user_warehouse_access_created_by_id",
        table_name="user_warehouse_access",
    )
    op.drop_table("user_warehouse_access")

    op.drop_index(
        "ix_user_permission_audit_warehouse_id",
        table_name="user_permission_audit",
    )
    op.drop_index(
        "ix_user_permission_audit_target_user_id",
        table_name="user_permission_audit",
    )
    op.drop_index(
        "ix_user_permission_audit_target_date",
        table_name="user_permission_audit",
    )
    op.drop_index(
        "ix_user_permission_audit_permission_code",
        table_name="user_permission_audit",
    )
    op.drop_index(
        "ix_user_permission_audit_created_at",
        table_name="user_permission_audit",
    )
    op.drop_index(
        "ix_user_permission_audit_actor_username",
        table_name="user_permission_audit",
    )
    op.drop_index(
        "ix_user_permission_audit_actor_user_id",
        table_name="user_permission_audit",
    )
    op.drop_index(
        "ix_user_permission_audit_actor_date",
        table_name="user_permission_audit",
    )
    op.drop_index(
        "ix_user_permission_audit_action",
        table_name="user_permission_audit",
    )
    op.drop_table("user_permission_audit")

    op.drop_index(
        "ix_user_permission_user_id",
        table_name="user_permission",
    )
    op.drop_index(
        "ix_user_permission_user",
        table_name="user_permission",
    )
    op.drop_index(
        "ix_user_permission_permission_id",
        table_name="user_permission",
    )
    op.drop_index(
        "ix_user_permission_granted_by_id",
        table_name="user_permission",
    )
    op.drop_table("user_permission")

    op.drop_index(
        "ix_warehouse_updated_by_id",
        table_name="warehouse",
    )
    op.drop_index(
        "ix_warehouse_name",
        table_name="warehouse",
    )
    op.drop_index(
        "ix_warehouse_is_active",
        table_name="warehouse",
    )
    op.drop_index(
        "ix_warehouse_created_by_id",
        table_name="warehouse",
    )
    op.drop_index(
        "ix_warehouse_code",
        table_name="warehouse",
    )
    op.drop_table("warehouse")

    op.drop_index(
        "ix_permission_is_active",
        table_name="permission",
    )
    op.drop_index(
        "ix_permission_group_name",
        table_name="permission",
    )
    op.drop_index(
        "ix_permission_group_active",
        table_name="permission",
    )
    op.drop_index(
        "ix_permission_code",
        table_name="permission",
    )
    op.drop_table("permission")
