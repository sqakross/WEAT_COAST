from alembic import op
import sqlalchemy as sa

# <<< ВАЖНО >>>
# Если твоя последняя миграция была revision="10132025" (из кода, который ты показал),
# то оставь down_revision = "10132025".
# revision снизу может быть любым новым уникальным ID.
revision = "102720251_applied_qty"
down_revision = "10132025"
branch_labels = None
depends_on = None


def upgrade():
    # Добавляем новую колонку applied_qty в таблицу goods_receipt_lines
    with op.batch_alter_table("goods_receipt_lines") as batch:
        batch.add_column(
            sa.Column(
                "applied_qty",
                sa.Integer(),
                nullable=False,
                server_default="0",  # временно нужно, чтобы БД разрешила NOT NULL
            )
        )

    # Убираем server_default, чтобы дальше база не проставляла 0 сама автоматически
    with op.batch_alter_table("goods_receipt_lines") as batch:
        batch.alter_column(
            "applied_qty",
            server_default=None,
        )


def downgrade():
    # Откат: просто удаляем колонку
    with op.batch_alter_table("goods_receipt_lines") as batch:
        batch.drop_column("applied_qty")
