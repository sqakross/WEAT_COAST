"""add issued_batch table + batch_id FK on issued_part_record"""

from alembic import op
import sqlalchemy as sa

# ⚠️ замените ниже revision на тот, что Alembic сгенерировал в имени файла
revision = 'b2c04705d903'
down_revision = "ffea2646ab9d"   # у тебя сейчас это текущая голова
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    insp = sa.inspect(bind)
    is_sqlite = bind.dialect.name == "sqlite"

    # 1) issued_batch — создаём, если её нет (create_all мог уже сделать)
    if not insp.has_table('issued_batch'):
        op.create_table(
            'issued_batch',
            sa.Column('id', sa.Integer, primary_key=True),
            sa.Column('invoice_number', sa.Integer, nullable=False, unique=True),
            sa.Column('issued_to', sa.String(255), nullable=False),
            sa.Column('issued_by', sa.String(255), nullable=False),
            sa.Column('reference_job', sa.String(255)),
            sa.Column('issue_date', sa.DateTime, nullable=False),
            sa.Column('location', sa.String(120)),
        )
        op.create_index('ix_issued_batch_invoice_number', 'issued_batch', ['invoice_number'], unique=True)
    else:
        # гарантируем индекс, если таблица уже была
        op.execute("CREATE UNIQUE INDEX IF NOT EXISTS ix_issued_batch_invoice_number ON issued_batch (invoice_number)")

    # 2) issued_part_record.batch_id — добавляем безопасно
    cols = {c['name'] for c in insp.get_columns('issued_part_record')}
    if 'batch_id' not in cols:
        if is_sqlite:
            # ✅ ВАЖНО: в SQLite — без batch_alter_table (чтобы не трогать view)
            op.add_column('issued_part_record', sa.Column('batch_id', sa.Integer(), nullable=True))
            op.create_index('ix_issued_part_record_batch_id', 'issued_part_record', ['batch_id'], unique=False)
            # FK в SQLite без пересоздания таблицы корректно не повесить — пропускаем
        else:
            # для других СУБД можно повесить FK
            with op.batch_alter_table("issued_part_record", schema=None) as batch_op:
                batch_op.add_column(sa.Column('batch_id', sa.Integer(), nullable=True))
                batch_op.create_index('ix_issued_part_record_batch_id', ['batch_id'], unique=False)
                batch_op.create_foreign_key(
                    'fk_issued_part_record_batch_id',
                    'issued_batch', ['batch_id'], ['id'],
                )
    else:
        # колонка уже есть — убедимся, что индекс есть
        op.execute("CREATE INDEX IF NOT EXISTS ix_issued_part_record_batch_id ON issued_part_record (batch_id)")

def downgrade():
    bind = op.get_bind()
    is_sqlite = bind.dialect.name == "sqlite"

    # issued_part_record: снимаем индекс/колонку
    try:
        if is_sqlite:
            op.execute("DROP INDEX IF EXISTS ix_issued_part_record_batch_id")
            # Удаление столбца в SQLite потребовало бы batch; если нужно — можно оставить как есть
            with op.batch_alter_table("issued_part_record", schema=None) as batch_op:
                batch_op.drop_column('batch_id')
        else:
            with op.batch_alter_table("issued_part_record", schema=None) as batch_op:
                try:
                    batch_op.drop_constraint('fk_issued_part_record_batch_id', type_='foreignkey')
                except Exception:
                    pass
                batch_op.drop_index('ix_issued_part_record_batch_id')
                batch_op.drop_column('batch_id')
    except Exception:
        pass

    # issued_batch: индекс + таблица
    try:
        op.drop_index('ix_issued_batch_invoice_number', table_name='issued_batch')
    except Exception:
        pass
    try:
        op.drop_table('issued_batch')
    except Exception:
        pass


