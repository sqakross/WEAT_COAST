"""
link supplier return to reports return

Revision ID: 20260414_link_supplier_return_to_reports
Revises: 20260411_add_retry_fields_to_email_outbox
Create Date: 2026-04-14
"""
from alembic import op
import sqlalchemy as sa


revision = "20260414_link_supplier_return_to_reports"
down_revision = "20260411_add_retry_fields_to_email_outbox"
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

    table = "supplier_return_batch"
    if not _table_exists(insp, table):
        return

    if not _column_exists(insp, table, "source_kind"):
        op.add_column(
            table,
            sa.Column("source_kind", sa.String(32), nullable=True)
        )

    if not _column_exists(insp, table, "source_return_record_id"):
        op.add_column(
            table,
            sa.Column("source_return_record_id", sa.Integer(), nullable=True)
        )

    # index для быстрого поиска
    try:
        op.create_index(
            "ix_supplier_return_source_record",
            table,
            ["source_kind", "source_return_record_id"]
        )
    except Exception:
        pass


def downgrade():
    bind = op.get_bind()
    insp = sa.inspect(bind)

    table = "supplier_return_batch"
    if not _table_exists(insp, table):
        return

    try:
        op.drop_index("ix_supplier_return_source_record", table_name=table)
    except Exception:
        pass

    if _column_exists(insp, table, "source_return_record_id"):
        op.drop_column(table, "source_return_record_id")

    if _column_exists(insp, table, "source_kind"):
        op.drop_column(table, "source_kind")