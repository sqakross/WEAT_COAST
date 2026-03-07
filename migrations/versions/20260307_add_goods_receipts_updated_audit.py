"""
add updated audit fields to goods_receipts

Revision ID: 20260307_add_goods_receipts_updated_audit
Revises: 20260219_add_receiving_lot_costs
Create Date: 2026-03-07
"""

from alembic import op
import sqlalchemy as sa

revision = "20260307_add_goods_receipts_updated_audit"
down_revision = "20260219_add_cost_link_to_issued_part_record"
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

    table = "goods_receipts"
    if not _table_exists(insp, table):
        return

    if not _column_exists(insp, table, "updated_at"):
        op.add_column(table, sa.Column("updated_at", sa.DateTime(), nullable=True))

    if not _column_exists(insp, table, "updated_by"):
        op.add_column(table, sa.Column("updated_by", sa.Integer(), nullable=True))

    if not _index_exists(insp, table, "ix_goods_receipts_updated_at"):
        try:
            op.create_index("ix_goods_receipts_updated_at", table, ["updated_at"], unique=False)
        except Exception:
            pass

    if not _index_exists(insp, table, "ix_goods_receipts_updated_by"):
        try:
            op.create_index("ix_goods_receipts_updated_by", table, ["updated_by"], unique=False)
        except Exception:
            pass

    if not _index_exists(insp, table, "ix_goods_receipts_created_by"):
        try:
            op.create_index("ix_goods_receipts_created_by", table, ["created_by"], unique=False)
        except Exception:
            pass


def downgrade():
    bind = op.get_bind()
    insp = sa.inspect(bind)

    table = "goods_receipts"
    if not _table_exists(insp, table):
        return

    for idx_name in [
        "ix_goods_receipts_updated_at",
        "ix_goods_receipts_updated_by",
        "ix_goods_receipts_created_by",
    ]:
        try:
            if _index_exists(insp, table, idx_name):
                op.drop_index(idx_name, table_name=table)
        except Exception:
            pass

    for col in ["updated_by", "updated_at"]:
        try:
            if _column_exists(insp, table, col):
                op.drop_column(table, col)
        except Exception:
            pass