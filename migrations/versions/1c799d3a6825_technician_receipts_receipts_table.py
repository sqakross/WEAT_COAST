"""technician receipts + receipts table (sqlite-safe)"""

from alembic import op
import sqlalchemy as sa

# ревизии
revision = "1c799d3a6825"
down_revision = None  # <-- оставь как у тебя; если Alembic поставил другое значение, СКИНЬ ЕГО НАЗАД
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    dialect = bind.dialect.name

    # --- ТАБЛИЦА issued_batch (если её ещё нет) ---
    # Alembic autogenerate обычно сам создаёт, но на всякий случай — create_if_not_exists-паттерн:
    try:
        op.create_table(
            "issued_batch",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("invoice_number", sa.Integer(), nullable=False),
            sa.Column("issued_to", sa.String(length=255), nullable=False),
            sa.Column("issued_by", sa.String(length=255), nullable=False),
            sa.Column("reference_job", sa.String(length=255)),
            sa.Column("issue_date", sa.DateTime(), nullable=False),
            sa.Column("location", sa.String(length=120)),
            sa.UniqueConstraint("invoice_number", name="uq_issued_batch_invoice_number"),
        )
        op.create_index("ix_issued_batch_invoice_number", "issued_batch", ["invoice_number"], unique=True)
    except Exception:
        # уже существует
        pass

    # --- issued_part_record: добавляем ссылки на batch, если их нет ---
    # колонка batch_id
    try:
        op.add_column("issued_part_record", sa.Column("batch_id", sa.Integer(), nullable=True))
    except Exception:
        pass
    # внешний ключ на issued_batch
    try:
        op.create_foreign_key(
            "fk_ipr_batch", "issued_part_record", "issued_batch",
            ["batch_id"], ["id"], ondelete="SET NULL"
        )
    except Exception:
        pass

    # --- изменения длины строковых полей (SAFE для SQLite) ---
    # SQLite НЕ поддерживает ALTER TYPE — пропускаем.
    if dialect != "sqlite":
        try:
            op.alter_column(
                "issued_part_record", "issued_to",
                existing_type=sa.String(length=100),
                type_=sa.String(length=255),
                existing_nullable=False,
            )
        except Exception:
            pass
        try:
            op.alter_column(
                "issued_part_record", "issued_by",
                existing_type=sa.String(length=100),
                type_=sa.String(length=255),
                existing_nullable=False,
            )
        except Exception:
            pass
        try:
            op.alter_column(
                "issued_part_record", "reference_job",
                existing_type=sa.String(length=100),
                type_=sa.String(length=255),
                existing_nullable=True,
            )
        except Exception:
            pass

    # --- WorkOrderPart: служебные поля (если их нет) ---
    # issued_qty
    try:
        op.add_column("work_order_parts", sa.Column("issued_qty", sa.Integer(), nullable=True))
    except Exception:
        pass
    # last_issued_at
    try:
        op.add_column("work_order_parts", sa.Column("last_issued_at", sa.DateTime(), nullable=True))
    except Exception:
        pass
    # stock_hint (для кеша подсказки склада)
    try:
        op.add_column("work_order_parts", sa.Column("stock_hint", sa.String(length=64), nullable=True))
    except Exception:
        pass

    # индексы (по желанию/если надо)
    try:
        op.create_index("ix_work_order_parts_work_order_id", "work_order_parts", ["work_order_id"])
    except Exception:
        pass
    try:
        op.create_index("ix_work_order_parts_unit_id", "work_order_parts", ["unit_id"])
    except Exception:
        pass


def downgrade():
    # откатываем по минимуму (безопасно)
    try:
        op.drop_constraint("fk_ipr_batch", "issued_part_record", type_="foreignkey")
    except Exception:
        pass
    try:
        op.drop_column("issued_part_record", "batch_id")
    except Exception:
        pass

    try:
        op.drop_index("ix_work_order_parts_work_order_id", table_name="work_order_parts")
    except Exception:
        pass
    try:
        op.drop_index("ix_work_order_parts_unit_id", table_name="work_order_parts")
    except Exception:
        pass

    try:
        op.drop_column("work_order_parts", "issued_qty")
    except Exception:
        pass
    try:
        op.drop_column("work_order_parts", "last_issued_at")
    except Exception:
        pass
    try:
        op.drop_column("work_order_parts", "stock_hint")
    except Exception:
        pass

    try:
        op.drop_index("ix_issued_batch_invoice_number", table_name="issued_batch")
    except Exception:
        pass
    try:
        op.drop_table("issued_batch")
    except Exception:
        pass
