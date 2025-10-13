from alembic import op
import sqlalchemy as sa

# Alembic identifiers
revision = "10122025"
down_revision = "48e9f313dce8"  # <-- поставь свою последнюю ревизию
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    insp = sa.inspect(bind)

    # goods_receipts
    tbls = set(insp.get_table_names())
    if 'goods_receipts' not in tbls:
        op.create_table(
            'goods_receipts',
            sa.Column('id', sa.Integer(), primary_key=True),
            sa.Column('supplier_name', sa.String(length=200), nullable=False),
            sa.Column('invoice_number', sa.String(length=64)),
            sa.Column('invoice_date', sa.Date()),
            sa.Column('currency', sa.String(length=8), nullable=False, server_default="USD"),
            sa.Column('notes', sa.Text()),
            sa.Column('status', sa.String(length=16), nullable=False, server_default="draft"),
            sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
            sa.Column('created_by', sa.Integer()),
            sa.Column('posted_at', sa.DateTime()),
            sa.Column('posted_by', sa.Integer()),
        )

    # индексы для goods_receipts
    existing = {ix['name'] for ix in insp.get_indexes('goods_receipts')} if 'goods_receipts' in tbls else set()
    if 'ix_goods_receipts_supplier' not in existing:
        op.create_index('ix_goods_receipts_supplier', 'goods_receipts', ['supplier_name'])
    if 'ix_gr_supplier_invoice' not in existing:
        op.create_index('ix_gr_supplier_invoice', 'goods_receipts', ['supplier_name', 'invoice_number'])

    # goods_receipt_lines
    tbls = set(insp.get_table_names())  # обновим
    if 'goods_receipt_lines' not in tbls:
        op.create_table(
            'goods_receipt_lines',
            sa.Column('id', sa.Integer(), primary_key=True),
            sa.Column('goods_receipt_id', sa.Integer(), sa.ForeignKey('goods_receipts.id', ondelete='CASCADE'), nullable=False),
            sa.Column('line_no', sa.Integer(), nullable=False, server_default="1"),
            sa.Column('part_number', sa.String(length=120), nullable=False),
            sa.Column('part_name', sa.String(length=255)),
            sa.Column('quantity', sa.Integer(), nullable=False, server_default="1"),
            sa.Column('unit_cost', sa.Float()),
            sa.Column('location', sa.String(length=64)),
        )

    existing = {ix['name'] for ix in insp.get_indexes('goods_receipt_lines')} if 'goods_receipt_lines' in tbls else set()
    if 'ix_grl_goods_receipt_id' not in existing:
        op.create_index('ix_grl_goods_receipt_id', 'goods_receipt_lines', ['goods_receipt_id'])
    if 'ix_grl_part_number' not in existing:
        op.create_index('ix_grl_part_number', 'goods_receipt_lines', ['part_number'])

def downgrade():
    # безопасный даунгрейд (удаляем индексы, затем таблицы — если есть)
    bind = op.get_bind()
    insp = sa.inspect(bind)

    if 'goods_receipt_lines' in insp.get_table_names():
        for name in ['ix_grl_part_number', 'ix_grl_goods_receipt_id']:
            try: op.drop_index(name, table_name='goods_receipt_lines')
            except Exception: pass
        op.drop_table('goods_receipt_lines')

    if 'goods_receipts' in insp.get_table_names():
        for name in ['ix_gr_supplier_invoice', 'ix_goods_receipts_supplier']:
            try: op.drop_index(name, table_name='goods_receipts')
            except Exception: pass
        op.drop_table('goods_receipts')