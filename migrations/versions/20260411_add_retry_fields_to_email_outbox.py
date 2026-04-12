"""
add retry fields to email_outbox

Revision ID: 20260411_add_retry_fields_to_email_outbox
Revises: 20260410_add_email_outbox_first_confirm
Create Date: 2026-04-11
"""

from alembic import op
import sqlalchemy as sa


revision = "20260411_add_retry_fields_to_email_outbox"
down_revision = "20260410_add_email_outbox_first_confirm"
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


def upgrade():
    bind = op.get_bind()
    insp = sa.inspect(bind)

    table = "email_outbox"
    if not _table_exists(insp, table):
        return

    if not _column_exists(insp, table, "attempt_count"):
        op.add_column(
            table,
            sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0")
        )

    if not _column_exists(insp, table, "last_attempt_at"):
        op.add_column(
            table,
            sa.Column("last_attempt_at", sa.DateTime(), nullable=True)
        )

    if not _column_exists(insp, table, "next_retry_at"):
        op.add_column(
            table,
            sa.Column("next_retry_at", sa.DateTime(), nullable=True)
        )


def downgrade():
    bind = op.get_bind()
    insp = sa.inspect(bind)

    table = "email_outbox"
    if not _table_exists(insp, table):
        return

    if _column_exists(insp, table, "next_retry_at"):
        op.drop_column(table, "next_retry_at")

    if _column_exists(insp, table, "last_attempt_at"):
        op.drop_column(table, "last_attempt_at")

    if _column_exists(insp, table, "attempt_count"):
        op.drop_column(table, "attempt_count")