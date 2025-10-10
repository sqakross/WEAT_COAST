"""add tech confirmation fields to issued_part_record

Revision ID: 20251009_add_confirm_fields
Revises: <PUT_PREV_REVISION_ID_HERE>
Create Date: 2025-10-09 00:00:00
"""

from alembic import op
import sqlalchemy as sa

revision = 'a2b60dbcddfd'
down_revision = 'fb55c44e6b4e'
branch_labels = None
depends_on = None

def upgrade():
    # 0) убрать хвосты от прерванных batch-операций
    op.execute("DROP TABLE IF EXISTS _alembic_tmp_issued_part_record")

    # 1) убрать зависимые VIEW, чтобы SQLite не валился на RENAME
    op.execute("DROP VIEW IF EXISTS v_work_order_parts_dto")
    op.execute("DROP VIEW IF EXISTS v_parts_search")

    # 2) добавить поля подтверждения
    with op.batch_alter_table('issued_part_record', schema=None) as batch:
        try:
            batch.add_column(sa.Column('confirmed_by_tech', sa.Boolean(), nullable=False, server_default=sa.text('false')))
        except Exception:
            batch.add_column(sa.Column('confirmed_by_tech', sa.Boolean(), nullable=False, server_default=sa.text('0')))
        batch.add_column(sa.Column('confirmed_at', sa.DateTime(), nullable=True))
        batch.add_column(sa.Column('confirmed_by', sa.String(length=64), nullable=True))
        batch.create_index('ix_issued_part_record_confirmed_by_tech', ['confirmed_by_tech'], unique=False)

    # 3) снять server_default по возможности
    try:
        op.alter_column('issued_part_record', 'confirmed_by_tech', server_default=None, existing_type=sa.Boolean())
    except Exception:
        pass

def downgrade():
    with op.batch_alter_table('issued_part_record', schema=None) as batch:
        try:
            batch.drop_index('ix_issued_part_record_confirmed_by_tech')
        except Exception:
            pass
        batch.drop_column('confirmed_by')
        batch.drop_column('confirmed_at')
        batch.drop_column('confirmed_by_tech')
    # VIEW назад не поднимаем — создадим отдельной ревизией


