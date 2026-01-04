from alembic import op
import sqlalchemy as sa

revision = "20260103_add_wo_audit_created_updated_by"
down_revision = "20251221_add_indexes_for_wo_and_issued"
branch_labels = None
depends_on = None


def _table_exists(insp, name: str) -> bool:
    try:
        return name in insp.get_table_names()
    except Exception:
        return False


def _col_exists(insp, table: str, col: str) -> bool:
    try:
        return col in [c["name"] for c in insp.get_columns(table)]
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

    # Your table is "work_orders" (per models.py)
    if _table_exists(insp, "work_orders"):

        # --- created_by_id ---
        if not _col_exists(insp, "work_orders", "created_by_id"):
            op.add_column("work_orders", sa.Column("created_by_id", sa.Integer(), nullable=True))
        if not _index_exists(insp, "work_orders", "ix_work_orders_created_by_id"):
            op.create_index("ix_work_orders_created_by_id", "work_orders", ["created_by_id"], unique=False)

        # --- updated_by_id ---
        if not _col_exists(insp, "work_orders", "updated_by_id"):
            op.add_column("work_orders", sa.Column("updated_by_id", sa.Integer(), nullable=True))
        if not _index_exists(insp, "work_orders", "ix_work_orders_updated_by_id"):
            op.create_index("ix_work_orders_updated_by_id", "work_orders", ["updated_by_id"], unique=False)

        # --- backfill for old rows (safe) ---
        # created_by_id = technician_id when empty
        if _col_exists(insp, "work_orders", "technician_id"):
            op.execute(
                sa.text("""
                    UPDATE work_orders
                    SET created_by_id = technician_id
                    WHERE created_by_id IS NULL AND technician_id IS NOT NULL
                """)
            )
            op.execute(
                sa.text("""
                    UPDATE work_orders
                    SET updated_by_id = technician_id
                    WHERE updated_by_id IS NULL AND technician_id IS NOT NULL
                """)
            )


def downgrade():
    bind = op.get_bind()
    insp = sa.inspect(bind)

    if _table_exists(insp, "work_orders"):

        if _index_exists(insp, "work_orders", "ix_work_orders_updated_by_id"):
            op.drop_index("ix_work_orders_updated_by_id", table_name="work_orders")
        if _col_exists(insp, "work_orders", "updated_by_id"):
            op.drop_column("work_orders", "updated_by_id")

        if _index_exists(insp, "work_orders", "ix_work_orders_created_by_id"):
            op.drop_index("ix_work_orders_created_by_id", table_name="work_orders")
        if _col_exists(insp, "work_orders", "created_by_id"):
            op.drop_column("work_orders", "created_by_id")
