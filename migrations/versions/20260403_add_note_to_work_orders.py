"""
add note to work_orders

Revision ID: 20260403_add_note_to_work_orders
Revises: 20260307_add_goods_receipts_updated_audit
Create Date: 2026-04-03
"""

from alembic import op
import sqlalchemy as sa

revision = "20260403_add_note_to_work_orders"
down_revision = "20260307_add_goods_receipts_updated_audit"
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

    table = "work_orders"
    if not _table_exists(insp, table):
        return

    if not _column_exists(insp, table, "note"):
        op.add_column(table, sa.Column("note", sa.Text(), nullable=True))


def downgrade():
    bind = op.get_bind()
    insp = sa.inspect(bind)

    table = "work_orders"
    if not _table_exists(insp, table):
        return

    if _column_exists(insp, table, "note"):
        op.drop_column(table, "note")