"""
add email outbox and first confirm email flag

Revision ID: 20260410_add_email_outbox_first_confirm
Revises: 20260403_add_note_to_work_orders
Create Date: 2026-04-10
"""

from alembic import op
import sqlalchemy as sa


revision = "20260410_add_email_outbox_first_confirm"
down_revision = "20260403_add_note_to_work_orders"
branch_labels = None
depends_on = None


def _table_exists(insp, name: str) -> bool:
    try:
        return name in insp.get_table_names()
    except Exception:
        return False


def _column_exists(insp, table: str, col: str) -> bool:
    try:
        cols = insp.get_columns(table) or []
        return any((c.get("name") == col) for c in cols)
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

    # -----------------------------
    # 1) email_outbox
    # -----------------------------
    if not _table_exists(insp, "email_outbox"):
        op.create_table(
            "email_outbox",
            sa.Column("id", sa.Integer(), primary_key=True),

            sa.Column("kind", sa.String(50), nullable=False),
            sa.Column("unique_key", sa.String(120), nullable=False),

            sa.Column("to_email", sa.String(255), nullable=False),
            sa.Column("subject", sa.String(255), nullable=False),
            sa.Column("body", sa.Text(), nullable=False),

            sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
            sa.Column("sent_at", sa.DateTime(), nullable=True),
            sa.Column("error", sa.Text(), nullable=True),

            sa.Column("work_order_id", sa.Integer(), nullable=True),
            sa.Column("batch_id", sa.Integer(), nullable=True),

            sa.ForeignKeyConstraint(["work_order_id"], ["work_orders.id"], ondelete="SET NULL"),
            sa.ForeignKeyConstraint(["batch_id"], ["issued_batch.id"], ondelete="SET NULL"),

            sa.UniqueConstraint("unique_key", name="uq_email_outbox_unique_key"),
        )

        op.create_index("ix_email_outbox_status", "email_outbox", ["status"])
        op.create_index("ix_email_outbox_work_order_id", "email_outbox", ["work_order_id"])
        op.create_index("ix_email_outbox_batch_id", "email_outbox", ["batch_id"])

    # refresh
    insp = sa.inspect(bind)

    # -----------------------------
    # 2) issued_batch flag
    # -----------------------------
    if _table_exists(insp, "issued_batch") and not _column_exists(insp, "issued_batch", "first_confirm_email_sent_at"):
        op.add_column(
            "issued_batch",
            sa.Column("first_confirm_email_sent_at", sa.DateTime(), nullable=True)
        )


def downgrade():
    bind = op.get_bind()
    insp = sa.inspect(bind)

    # -----------------------------
    # remove column
    # -----------------------------
    if _table_exists(insp, "issued_batch") and _column_exists(insp, "issued_batch", "first_confirm_email_sent_at"):
        op.drop_column("issued_batch", "first_confirm_email_sent_at")

    insp = sa.inspect(bind)

    # -----------------------------
    # drop email_outbox
    # -----------------------------
    if _table_exists(insp, "email_outbox"):
        if _index_exists(insp, "email_outbox", "ix_email_outbox_batch_id"):
            op.drop_index("ix_email_outbox_batch_id", table_name="email_outbox")
        if _index_exists(insp, "email_outbox", "ix_email_outbox_work_order_id"):
            op.drop_index("ix_email_outbox_work_order_id", table_name="email_outbox")
        if _index_exists(insp, "email_outbox", "ix_email_outbox_status"):
            op.drop_index("ix_email_outbox_status", table_name="email_outbox")

        op.drop_table("email_outbox")