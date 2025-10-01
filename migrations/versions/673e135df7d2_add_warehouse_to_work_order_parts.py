"""add warehouse to work_order_parts

Revision ID: 673e135df7d2
Revises: XXXXXXXX
Create Date: 2025-09-24 14:31:56.235663
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '673e135df7d2'
down_revision = None   # ВАЖНО: привязываем к твоей ревизии XXXXXXXX
branch_labels = None
depends_on = None


def upgrade():
    # 1) Добавляем колонку warehouse (SQLite дружелюбно)
    op.add_column('work_order_parts', sa.Column('warehouse', sa.String(length=120), nullable=True))

    # 2) Бэкофилл: переносим из unit_label, если warehouse пуст
    conn = op.get_bind()
    conn.execute(sa.text("""
        UPDATE work_order_parts
        SET warehouse = unit_label
        WHERE warehouse IS NULL OR warehouse = ''
    """))

def downgrade():
    # Откат: удалить колонку
    op.drop_column('work_order_parts', 'warehouse')

