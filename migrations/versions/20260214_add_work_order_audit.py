"""add work_order_audit table

Revision ID: 20260214_add_work_order_audit
Revises: 20260201_add_job_reservation
Create Date: 2026-02-14
"""

from alembic import op
import sqlalchemy as sa

revision = "20260214_add_work_order_audit"
down_revision = "20260201_add_job_reservation"
branch_labels = None
depends_on = None


def _table_exists(insp, name: str) -> bool:
    try:
        return name in insp.get_table_names()
    except Exception:
        return False


def _index_exists(insp, table: str, index_name: str) -> bool:
    try:
        idxs = insp.get_indexes(table) or []
        return any((i.get("name") == index_name) for i in idxs)
    except Exception:
        return False


def upgrade():
    bind = op.get_bind()
    insp = sa.inspect(bind)

    # 1) create work_order_audit table
    if not _table_exists(insp, "work_order_audit"):
        op.create_table(
            "work_order_audit",
            sa.Column("id", sa.Integer(), primary_key=True),

            sa.Column("work_order_id", sa.Integer(), nullable=False),
            sa.Column("action", sa.String(length=40), nullable=False),
            sa.Column("message", sa.String(length=255), nullable=False),
            sa.Column("meta_json", sa.Text(), nullable=True),

            sa.Column("actor_user_id", sa.Integer(), nullable=True),
            sa.Column("actor_username", sa.String(length=64), nullable=True),

            # SQLite-friendly timestamp default
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),

            sa.ForeignKeyConstraint(["work_order_id"], ["work_orders.id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["actor_user_id"], ["user.id"]),
        )

    # 2) indexes (idempotent)
    if _table_exists(insp, "work_order_audit"):
        if not _index_exists(insp, "work_order_audit", "ix_work_order_audit_work_order_id"):
            op.create_index("ix_work_order_audit_work_order_id", "work_order_audit", ["work_order_id"], unique=False)

        if not _index_exists(insp, "work_order_audit", "ix_work_order_audit_created_at"):
            op.create_index("ix_work_order_audit_created_at", "work_order_audit", ["created_at"], unique=False)

        if not _index_exists(insp, "work_order_audit", "ix_work_order_audit_action"):
            op.create_index("ix_work_order_audit_action", "work_order_audit", ["action"], unique=False)

        if not _index_exists(insp, "work_order_audit", "ix_work_order_audit_actor_user_id"):
            op.create_index("ix_work_order_audit_actor_user_id", "work_order_audit", ["actor_user_id"], unique=False)


def downgrade():
    bind = op.get_bind()
    insp = sa.inspect(bind)

    if _table_exists(insp, "work_order_audit"):
        # drop indexes (if exist)
        for idx in [
            "ix_work_order_audit_actor_user_id",
            "ix_work_order_audit_action",
            "ix_work_order_audit_created_at",
            "ix_work_order_audit_work_order_id",
        ]:
            try:
                if _index_exists(insp, "work_order_audit", idx):
                    op.drop_index(idx, table_name="work_order_audit")
            except Exception:
                pass

        op.drop_table("work_order_audit")
