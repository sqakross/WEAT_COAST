"""add ordered_date (auto-detect table name)"""

from alembic import op
import sqlalchemy as sa

revision = "20251010_add_ordered_date_autodetect"
down_revision = "dccaf46c9bac"   # твоя текущая HEAD
branch_labels = None
depends_on = None


def _resolve_table_name(conn) -> str:
    """
    Аккуратно выясняем реальное имя таблицы WorkOrderPart.
    Пробуем несколько самых вероятных вариантов и также сверяемся с метаданными.
    """
    inspector = sa.inspect(conn)
    tables = set(inspector.get_table_names())

    # Часто встречающиеся варианты
    candidates = [
        "work_order_part",
        "work_order_parts",
        "workorder_part",
        "workorder_parts",
        "work_order_part_tbl",
    ]
    for name in candidates:
        if name in tables:
            return name

    # Если не нашли — подсказываем какие таблицы есть
    raise RuntimeError(
        "Не удалось найти таблицу WorkOrderPart. Доступные таблицы: "
        + ", ".join(sorted(tables))
    )


def _column_exists(conn, table: str, column: str) -> bool:
    inspector = sa.inspect(conn)
    cols = [c["name"] for c in inspector.get_columns(table)]
    return column in cols


def upgrade():
    conn = op.get_bind()
    table = _resolve_table_name(conn)

    # Ничего не делаем, если колонка уже есть (повторный запуск миграции будет безопасным)
    if _column_exists(conn, table, "ordered_date"):
        return

    # Для SQLite нормальный ALTER ADD COLUMN
    with op.batch_alter_table(table, schema=None) as batch:
        batch.add_column(sa.Column("ordered_date", sa.Date(), nullable=True))


def downgrade():
    conn = op.get_bind()
    try:
        table = _resolve_table_name(conn)
    except Exception:
        # Если даже таблицу не находим — тихо выходим
        return

    if not _column_exists(conn, table, "ordered_date"):
        return

    with op.batch_alter_table(table, schema=None) as batch:
        batch.drop_column("ordered_date")
