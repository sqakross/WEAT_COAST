"""
add lot/base/actual costs to goods_receipt_lines

Revision ID: 20260219_add_receiving_lot_costs
Revises: 20260214_add_work_order_audit
Create Date: 2026-02-19
"""

from alembic import op
import sqlalchemy as sa

revision = "20260219_add_receiving_lot_costs"
down_revision = "20260214_add_work_order_audit"
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

    table = "goods_receipt_lines"
    if not _table_exists(insp, table):
        # у тебя алиасы ReceivingItem=GoodsReceiptLine, но таблица должна быть именно эта
        return

    # --- 1) add columns (idempotent) ---
    if not _column_exists(insp, table, "base_unit_cost"):
        op.add_column(table, sa.Column("base_unit_cost", sa.Float(), nullable=True))

    if not _column_exists(insp, table, "extra_alloc_per_unit"):
        # server_default важен, чтобы новые записи не имели NULL в расчетах
        op.add_column(
            table,
            sa.Column(
                "extra_alloc_per_unit",
                sa.Float(),
                nullable=True,
                server_default=sa.text("0"),
            ),
        )

    if not _column_exists(insp, table, "actual_unit_cost"):
        op.add_column(table, sa.Column("actual_unit_cost", sa.Float(), nullable=True))

    # --- 2) helpful index for lot lookup: (goods_receipt_id, part_number) ---
    # SQLite: create_index ok, idempotent check
    idx_name = "ix_grl_receipt_pn"
    if not _index_exists(insp, table, idx_name):
        try:
            op.create_index(idx_name, table, ["goods_receipt_id", "part_number"], unique=False)
        except Exception:
            # если в конкретной БД индекс уже есть под другим именем — не падаем
            pass


def downgrade():
    bind = op.get_bind()
    insp = sa.inspect(bind)

    table = "goods_receipt_lines"
    if not _table_exists(insp, table):
        return

    # drop index first
    idx_name = "ix_grl_receipt_pn"
    try:
        if _index_exists(insp, table, idx_name):
            op.drop_index(idx_name, table_name=table)
    except Exception:
        pass

    # drop columns (SQLite supports DROP COLUMN only in newer versions; alembic may emulate poorly)
    # To stay safe on SQLite, we do a soft-downgrade: try, but ignore failures.
    for col in ["actual_unit_cost", "extra_alloc_per_unit", "base_unit_cost"]:
        try:
            if _column_exists(insp, table, col):
                op.drop_column(table, col)
        except Exception:
            pass
