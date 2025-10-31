from alembic import op
import sqlalchemy as sa

# Уникальный ID для новой миграции
revision = "20251030_extra_expenses"
down_revision = "102720251_applied_qty"  # <-- ПРОШЛАЯ миграция из твоего примера
branch_labels = None
depends_on = None


def upgrade():
    # Добавляем колонку extra_expenses в goods_receipts
    #
    # Логика такая же, как мы делали с applied_qty:
    # 1. Сначала добавляем колонку с server_default='0', чтобы база разрешила NOT NULL даже на существующих строках.
    # 2. Потом убираем server_default, чтобы дальше 0 не ставился автоматически.
    with op.batch_alter_table("goods_receipts") as batch:
        batch.add_column(
            sa.Column(
                "extra_expenses",
                sa.Float(),
                nullable=False,
                server_default="0",
            )
        )

    # Убираем server_default после инициализации
    with op.batch_alter_table("goods_receipts") as batch:
        batch.alter_column(
            "extra_expenses",
            server_default=None,
        )


def downgrade():
    # Откат: удаляем колонку extra_expenses
    with op.batch_alter_table("goods_receipts") as batch:
        batch.drop_column("extra_expenses")
