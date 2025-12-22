from alembic import op
import sqlalchemy as sa

revision = "20251221_add_indexes_for_wo_and_issued"
down_revision = "20251215_add_inv_ref_to_issued_part_record"
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

    # ------------------------------------------------------------
    # issued_batch: индекс на reference_job (для точного поиска)
    # ------------------------------------------------------------
    if _table_exists(insp, "issued_batch") and _col_exists(insp, "issued_batch", "reference_job"):
        if not _index_exists(insp, "issued_batch", "ix_issued_batch_reference_job"):
            op.create_index(
                "ix_issued_batch_reference_job",
                "issued_batch",
                ["reference_job"],
                unique=False,
            )

    # ------------------------------------------------------------
    # issued_part_record: индекс на reference_job (если нет)
    # (обычно он уже есть, но проверим)
    # ------------------------------------------------------------
    if _table_exists(insp, "issued_part_record") and _col_exists(insp, "issued_part_record", "reference_job"):
        if not _index_exists(insp, "issued_part_record", "ix_issued_part_record_reference_job"):
            op.create_index(
                "ix_issued_part_record_reference_job",
                "issued_part_record",
                ["reference_job"],
                unique=False,
            )

    # ------------------------------------------------------------
    # issued_part_record: индекс на batch_id (ускоряет join + фильтры)
    # ------------------------------------------------------------
    if _table_exists(insp, "issued_part_record") and _col_exists(insp, "issued_part_record", "batch_id"):
        if not _index_exists(insp, "issued_part_record", "ix_issued_part_record_batch_id"):
            op.create_index(
                "ix_issued_part_record_batch_id",
                "issued_part_record",
                ["batch_id"],
                unique=False,
            )

    # ------------------------------------------------------------
    # work_order: индекс на canonical_job (для точного маппинга job -> WO)
    # ------------------------------------------------------------
    if _table_exists(insp, "work_order") and _col_exists(insp, "work_order", "canonical_job"):
        if not _index_exists(insp, "work_order", "ix_work_order_canonical_job"):
            op.create_index(
                "ix_work_order_canonical_job",
                "work_order",
                ["canonical_job"],
                unique=False,
            )

    # ------------------------------------------------------------
    # work_order_part: индексы для быстрых поисков по PN / invoice_number
    # ------------------------------------------------------------
    if _table_exists(insp, "work_order_part"):
        if _col_exists(insp, "work_order_part", "part_number") and not _index_exists(insp, "work_order_part", "ix_work_order_part_part_number"):
            op.create_index(
                "ix_work_order_part_part_number",
                "work_order_part",
                ["part_number"],
                unique=False,
            )

        if _col_exists(insp, "work_order_part", "invoice_number") and not _index_exists(insp, "work_order_part", "ix_work_order_part_invoice_number"):
            op.create_index(
                "ix_work_order_part_invoice_number",
                "work_order_part",
                ["invoice_number"],
                unique=False,
            )


def downgrade():
    bind = op.get_bind()
    insp = sa.inspect(bind)

    # В downgrade аккуратно: удаляем только те индексы, которые мы создавали
    if _table_exists(insp, "work_order_part"):
        if _index_exists(insp, "work_order_part", "ix_work_order_part_invoice_number"):
            op.drop_index("ix_work_order_part_invoice_number", table_name="work_order_part")
        if _index_exists(insp, "work_order_part", "ix_work_order_part_part_number"):
            op.drop_index("ix_work_order_part_part_number", table_name="work_order_part")

    if _table_exists(insp, "work_order"):
        if _index_exists(insp, "work_order", "ix_work_order_canonical_job"):
            op.drop_index("ix_work_order_canonical_job", table_name="work_order")

    if _table_exists(insp, "issued_part_record"):
        if _index_exists(insp, "issued_part_record", "ix_issued_part_record_batch_id"):
            op.drop_index("ix_issued_part_record_batch_id", table_name="issued_part_record")
        if _index_exists(insp, "issued_part_record", "ix_issued_part_record_reference_job"):
            op.drop_index("ix_issued_part_record_reference_job", table_name="issued_part_record")

    if _table_exists(insp, "issued_batch"):
        if _index_exists(insp, "issued_batch", "ix_issued_batch_reference_job"):
            op.drop_index("ix_issued_batch_reference_job", table_name="issued_batch")
