"""Add stock and consumption tracking fields (safe, checks table names)"""

from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = "11072025_add_stock"
down_revision = "20251030_extra_expenses"
branch_labels = None
depends_on = None


def _get_existing_tables():
    bind = op.get_bind()
    insp = sa.inspect(bind)
    try:
        return set(insp.get_table_names())
    except Exception:
        return set()


def _add_col_if_exists(table_name: str, column: sa.Column):
    tables = _get_existing_tables()
    if table_name in tables:
        op.add_column(table_name, column)


def _try_exec(sql: str):
    try:
        op.execute(sql)
    except Exception:
        pass


def upgrade():
    tables = _get_existing_tables()

    # --- кандидаты имён по историческим версиям ---
    part_tables = []
    if "issued_part_records" in tables:
        part_tables.append("issued_part_records")
    if "issued_part_record" in tables:
        part_tables.append("issued_part_record")

    batch_tables = []
    if "issued_batches" in tables:
        batch_tables.append("issued_batches")
    if "issued_batch" in tables:
        batch_tables.append("issued_batch")

    # === строки выдачи (issued_part_record[s]) ===
    for t in part_tables:
        _add_col_if_exists(t, sa.Column("consumed_qty", sa.Integer(), nullable=True))
        _add_col_if_exists(t, sa.Column("consumed_flag", sa.Boolean(), nullable=False, server_default=sa.false()))
        _add_col_if_exists(t, sa.Column("consumed_at", sa.DateTime(), nullable=True))
        _add_col_if_exists(t, sa.Column("consumed_by", sa.String(length=120), nullable=True))
        _add_col_if_exists(t, sa.Column("consumed_note", sa.String(length=500), nullable=True))

    # === заголовки выдачи (issued_batch[es]) ===
    for t in batch_tables:
        _add_col_if_exists(t, sa.Column("is_stock", sa.Boolean(), nullable=False, server_default=sa.false()))
        _add_col_if_exists(t, sa.Column("consumed_flag", sa.Boolean(), nullable=False, server_default=sa.false()))
        _add_col_if_exists(t, sa.Column("consumed_at", sa.DateTime(), nullable=True))
        _add_col_if_exists(t, sa.Column("consumed_by", sa.String(length=120), nullable=True))
        _add_col_if_exists(t, sa.Column("consumed_note", sa.String(length=500), nullable=True))

    # --- Опционально: авто-проставить is_stock для старых батчей по reference_job LIKE 'stock%' ---
    # Пытаемся для обоих вариантов имён и обоих вариантов имени столбца reference_job
    for t in batch_tables:
        # Попробуем самые типичные имена столбцов
        _try_exec(f"""
            UPDATE {t}
               SET is_stock = 1
             WHERE reference_job IS NOT NULL
               AND LOWER(reference_job) LIKE 'stock%';
        """)

        # Иногда поле могло называться ref или invoice_ref — не критично, пропускаем, если нет


def downgrade():
    tables = _get_existing_tables()

    # кандидаты имён
    part_tables = []
    if "issued_part_records" in tables:
        part_tables.append("issued_part_records")
    if "issued_part_record" in tables:
        part_tables.append("issued_part_record")

    batch_tables = []
    if "issued_batches" in tables:
        batch_tables.append("issued_batches")
    if "issued_batch" in tables:
        batch_tables.append("issued_batch")

    # issued_part_record[s]
    for t in part_tables:
        try:
            op.drop_column(t, "consumed_note")
        except Exception:
            pass
        try:
            op.drop_column(t, "consumed_by")
        except Exception:
            pass
        try:
            op.drop_column(t, "consumed_at")
        except Exception:
            pass
        try:
            op.drop_column(t, "consumed_flag")
        except Exception:
            pass
        try:
            op.drop_column(t, "consumed_qty")
        except Exception:
            pass

    # issued_batch[es]
    for t in batch_tables:
        try:
            op.drop_column(t, "consumed_note")
        except Exception:
            pass
        try:
            op.drop_column(t, "consumed_by")
        except Exception:
            pass
        try:
            op.drop_column(t, "consumed_at")
        except Exception:
            pass
        try:
            op.drop_column(t, "consumed_flag")
        except Exception:
            pass
        try:
            op.drop_column(t, "is_stock")
        except Exception:
            pass
