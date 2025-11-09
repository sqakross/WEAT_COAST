"""Supplier Returns: batches + items (safe create with existence checks)"""

from alembic import op
import sqlalchemy as sa

# --- ревизии ---
revision = "20251108_supplier_returns"
down_revision = "11072025_add_stock"
branch_labels = None
depends_on = None


# ===== ВСПОМОГАТЕЛЬНОЕ =====
def _get_existing_tables():
    bind = op.get_bind()
    insp = sa.inspect(bind)
    try:
        return set(insp.get_table_names())
    except Exception:
        return set()


def _table_exists(name: str) -> bool:
    return name in _get_existing_tables()


def _create_table_if_absent(name: str, columns: list[sa.Column], fks: list[sa.schema.ForeignKeyConstraint] | None = None, indexes: list[tuple[str, list[str]]] | None = None):
    """
    Безопасно создаёт таблицу, если её нет.
    indexes: список кортежей (имя_индекса, [колонки])
    """
    if _table_exists(name):
        return

    args = [*columns]
    if fks:
        args.extend(fks)

    op.create_table(name, *args)

    # Индексы (после create_table)
    if indexes:
        for ix_name, cols in indexes:
            try:
                op.create_index(ix_name, name, cols)
            except Exception:
                pass


def upgrade():
    # --- 1) supplier_return_batch ---
    _create_table_if_absent(
        "supplier_return_batch",
        columns=[
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("supplier_name", sa.String(length=200), nullable=True, index=False),
            sa.Column("reference_receiving_id", sa.Integer(), nullable=True),  # опционально, связи не навязываем
            sa.Column("status", sa.String(length=20), nullable=False, server_default="draft"),  # draft|posted
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.Column("created_by", sa.String(length=120), nullable=True),
            sa.Column("posted_at", sa.DateTime(), nullable=True),
            sa.Column("posted_by", sa.String(length=120), nullable=True),

            sa.Column("total_items", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("total_value", sa.Float(),   nullable=False, server_default="0"),
        ],
        fks=None,
        indexes=[
            ("ix_srb_status", ["status"]),
            ("ix_srb_supplier", ["supplier_name"]),
        ],
    )

    # --- 2) supplier_return_item ---
    fk = sa.ForeignKeyConstraint(
        ["batch_id"], ["supplier_return_batch.id"], name="fk_sri_batch", ondelete="CASCADE"
    )

    _create_table_if_absent(
        "supplier_return_item",
        columns=[
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("batch_id", sa.Integer(), nullable=False),
            sa.Column("part_number", sa.String(length=120), nullable=False),
            sa.Column("part_name",   sa.String(length=255), nullable=True),
            sa.Column("location",    sa.String(length=120), nullable=True),

            sa.Column("qty_returned", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("unit_cost",    sa.Float(),   nullable=False, server_default="0"),
            sa.Column("total_cost",   sa.Float(),   nullable=False, server_default="0"),
        ],
        fks=[fk],
        indexes=[
            ("ix_sri_part_number", ["part_number"]),
            ("ix_sri_batch", ["batch_id"]),
            ("ix_sri_location", ["location"]),
        ],
    )

    # --- на всякий случай: убедимся в существовании индексов (если таблицы уже были) ---
    # (Alembic не имеет "create_index_if_not_exists"; тихо пробуем)
    try:
        op.create_index("ix_srb_status", "supplier_return_batch", ["status"])
    except Exception:
        pass
    try:
        op.create_index("ix_srb_supplier", "supplier_return_batch", ["supplier_name"])
    except Exception:
        pass
    try:
        op.create_index("ix_sri_part_number", "supplier_return_item", ["part_number"])
    except Exception:
        pass
    try:
        op.create_index("ix_sri_batch", "supplier_return_item", ["batch_id"])
    except Exception:
        pass
    try:
        op.create_index("ix_sri_location", "supplier_return_item", ["location"])
    except Exception:
        pass


def downgrade():
    # удаляем в порядке зависимостей
    try:
        op.drop_table("supplier_return_item")
    except Exception:
        pass
    try:
        op.drop_table("supplier_return_batch")
    except Exception:
        pass
