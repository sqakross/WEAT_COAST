from alembic import op
import sqlalchemy as sa

# --- ревизии ---
revision = "20251124_supplier_return_tech_note"
down_revision = "20251108_supplier_returns"  # <-- твоя предыдущая миграция
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


def _column_exists(table_name: str, column_name: str) -> bool:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    try:
        cols = [c["name"] for c in insp.get_columns(table_name)]
        return column_name in cols
    except Exception:
        return False


def upgrade():
    # добавляем колонку tech_note, если таблица есть и колонки ещё нет
    if _table_exists("supplier_return_batch") and not _column_exists("supplier_return_batch", "tech_note"):
        with op.batch_alter_table("supplier_return_batch") as batch_op:
            batch_op.add_column(
                sa.Column("tech_note", sa.String(length=255), nullable=True)
            )


def downgrade():
    # аккуратно удаляем колонку при откате
    if _table_exists("supplier_return_batch") and _column_exists("supplier_return_batch", "tech_note"):
        with op.batch_alter_table("supplier_return_batch") as batch_op:
            batch_op.drop_column("tech_note")
