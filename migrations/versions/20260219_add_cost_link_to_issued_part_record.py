"""add cost link fields to issued_part_record

Revision ID: 20260219_add_cost_link_to_issued_part_record
Revises: 20260214_add_work_order_audit
Create Date: 2026-02-19
"""

from alembic import op
import sqlalchemy as sa

revision = "20260219_add_cost_link_to_issued_part_record"
down_revision = "20260219_add_receiving_lot_costs"
branch_labels = None
depends_on = None


def _table_exists(insp, name: str) -> bool:
    try:
        return name in insp.get_table_names()
    except Exception:
        return False


def _col_exists(insp, table: str, col: str) -> bool:
    try:
        cols = insp.get_columns(table) or []
        return any(c.get("name") == col for c in cols)
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

    if not _table_exists(insp, "issued_part_record"):
        return

    # --- add columns (SQLite-safe) ---
    if not _col_exists(insp, "issued_part_record", "source_receipt_line_id"):
        op.add_column(
            "issued_part_record",
            sa.Column("source_receipt_line_id", sa.Integer(), nullable=True),
        )

    if not _col_exists(insp, "issued_part_record", "cost_source"):
        op.add_column(
            "issued_part_record",
            sa.Column("cost_source", sa.String(length=32), nullable=True),
        )

    # --- indexes ---
    if not _index_exists(insp, "issued_part_record", "ix_issued_part_record_source_receipt_line_id"):
        op.create_index(
            "ix_issued_part_record_source_receipt_line_id",
            "issued_part_record",
            ["source_receipt_line_id"],
            unique=False,
        )

    if not _index_exists(insp, "issued_part_record", "ix_issued_part_record_cost_source"):
        op.create_index(
            "ix_issued_part_record_cost_source",
            "issued_part_record",
            ["cost_source"],
            unique=False,
        )

    # NOTE: FK is intentionally skipped for SQLite safety (can't reliably add FK after create).


def downgrade():
    bind = op.get_bind()
    insp = sa.inspect(bind)

    if not _table_exists(insp, "issued_part_record"):
        return

    # drop indexes first
    for idx in [
        "ix_issued_part_record_cost_source",
        "ix_issued_part_record_source_receipt_line_id",
    ]:
        try:
            if _index_exists(insp, "issued_part_record", idx):
                op.drop_index(idx, table_name="issued_part_record")
        except Exception:
            pass

    # drop columns (if your SQLite env doesn't support drop_column, it's ok to leave this)
    try:
        if _col_exists(insp, "issued_part_record", "cost_source"):
            op.drop_column("issued_part_record", "cost_source")
    except Exception:
        pass

    try:
        if _col_exists(insp, "issued_part_record", "source_receipt_line_id"):
            op.drop_column("issued_part_record", "source_receipt_line_id")
    except Exception:
        pass
