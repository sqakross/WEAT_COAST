"""Add goods_receipts and goods_receipt_lines"""

from alembic import op
import sqlalchemy as sa

# Идентификаторы ревизий
revision = "48e9f313dce8"      # Alembic подставит сам, оставь как есть
down_revision = "da0dc262ace1"               # ВАЖНО: ставим твою последнюю ревизию
branch_labels = None
depends_on = None


def upgrade():
    # Таблица шапок приходов
    op.create_table(
        'goods_receipts',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('supplier_name', sa.String(length=200), nullable=False),
        sa.Column('invoice_number', sa.String(length=64), nullable=True),
        sa.Column('invoice_date', sa.Date(), nullable=True),
        sa.Column('currency', sa.String(length=8), nullable=False, server_default="USD"),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('status', sa.String(length=16), nullable=False, server_default="draft"),  # draft|posted
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('created_by', sa.Integer(), nullable=True),
        sa.Column('posted_at', sa.DateTime(), nullable=True),
        sa.Column('posted_by', sa.Integer(), nullable=True),
    )
    # Индексы для быстрого поиска
    op.create_index('ix_goods_receipts_supplier', 'goods_receipts', ['supplier_name'])
    op.create_index('ix_gr_supplier_invoice', 'goods_receipts', ['supplier_name', 'invoice_number'])

    # Таблица строк приходов
    op.create_table(
        'goods_receipt_lines',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('goods_receipt_id', sa.Integer(), sa.ForeignKey('goods_receipts.id', ondelete='CASCADE'), nullable=False),
        sa.Column('line_no', sa.Integer(), nullable=False, server_default="1"),
        sa.Column('part_number', sa.String(length=120), nullable=False),
        sa.Column('part_name', sa.String(length=255), nullable=True),
        sa.Column('quantity', sa.Integer(), nullable=False, server_default="1"),
        sa.Column('unit_cost', sa.Float(), nullable=True),
        sa.Column('location', sa.String(length=64), nullable=True),
    )
    # Индексы для строк
    op.create_index('ix_grl_goods_receipt_id', 'goods_receipt_lines', ['goods_receipt_id'])
    op.create_index('ix_grl_part_number', 'goods_receipt_lines', ['part_number'])


def downgrade():
    op.drop_index('ix_grl_part_number', table_name='goods_receipt_lines')
    op.drop_index('ix_grl_goods_receipt_id', table_name='goods_receipt_lines')
    op.drop_table('goods_receipt_lines')

    op.drop_index('ix_gr_supplier_invoice', table_name='goods_receipts')
    op.drop_index('ix_goods_receipts_supplier', table_name='goods_receipts')
    op.drop_table('goods_receipts')
