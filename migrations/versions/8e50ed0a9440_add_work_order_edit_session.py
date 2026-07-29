"""Add work order edit session

Revision ID: 8e50ed0a9440
Revises: 9f2c1a7b4d10
Create Date: 2026-07-29 08:43:35.515552
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "8e50ed0a9440"
down_revision = "9f2c1a7b4d10"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "work_order_edit_session",

        sa.Column(
            "id",
            sa.Integer(),
            nullable=False,
        ),

        sa.Column(
            "work_order_id",
            sa.Integer(),
            nullable=False,
        ),

        sa.Column(
            "user_id",
            sa.Integer(),
            nullable=False,
        ),

        sa.Column(
            "username",
            sa.String(length=64),
            nullable=False,
        ),

        sa.Column(
            "opened_at",
            sa.DateTime(),
            nullable=False,
        ),

        sa.Column(
            "last_seen_at",
            sa.DateTime(),
            nullable=False,
        ),

        sa.ForeignKeyConstraint(
            ["work_order_id"],
            ["work_orders.id"],
            ondelete="CASCADE",
        ),

        sa.ForeignKeyConstraint(
            ["user_id"],
            ["user.id"],
            ondelete="CASCADE",
        ),

        sa.PrimaryKeyConstraint("id"),

        sa.UniqueConstraint(
            "work_order_id",
            name="uq_work_order_edit_session_work_order_id",
        ),
    )

    op.create_index(
        "ix_work_order_edit_session_work_order_id",
        "work_order_edit_session",
        ["work_order_id"],
        unique=False,
    )

    op.create_index(
        "ix_work_order_edit_session_user_id",
        "work_order_edit_session",
        ["user_id"],
        unique=False,
    )

    op.create_index(
        "ix_work_order_edit_session_last_seen_at",
        "work_order_edit_session",
        ["last_seen_at"],
        unique=False,
    )


def downgrade():
    op.drop_index(
        "ix_work_order_edit_session_last_seen_at",
        table_name="work_order_edit_session",
    )

    op.drop_index(
        "ix_work_order_edit_session_user_id",
        table_name="work_order_edit_session",
    )

    op.drop_index(
        "ix_work_order_edit_session_work_order_id",
        table_name="work_order_edit_session",
    )

    op.drop_table(
        "work_order_edit_session"
    )