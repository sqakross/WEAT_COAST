# migrations/versions/<stamp>_add_technician_id_to_work_orders.py
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "da0dc262ace1"
down_revision = "20251010_add_ordered_date_autodetect"
branch_labels = None
depends_on = None


def upgrade():
    # 1) Добавляем колонку technician_id (nullable=True для мягкого перехода)
    # batch_alter_table — дружит с SQLite
    with op.batch_alter_table("work_orders", schema=None) as batch_op:
        batch_op.add_column(sa.Column("technician_id", sa.Integer(), nullable=True))
        batch_op.create_index("ix_work_orders_technician_id", ["technician_id"], unique=False)
        # Попытка создать FK (если не SQLite — применится; в SQLite Alembic сам эмулирует)
        try:
            batch_op.create_foreign_key(
                "fk_work_orders_technician_id_user",
                "user",  # имя таблицы User по умолчанию = 'user'
                ["technician_id"],
                ["id"],
                ondelete=None,
            )
        except Exception:
            # На старых SQLite может не получиться — не критично
            pass

    # 2) Бэкофисная подзаправка technician_id по совпадению имени
    #    Нормализуем: TRIM + нижний регистр
    conn = op.get_bind()

    # Построим карту username(lower)->id только для role='technician'
    user_rows = conn.execute(sa.text("""
        SELECT id, LOWER(TRIM(username)) AS uname
        FROM "user"
        WHERE LOWER(TRIM(role)) = 'technician'
    """)).fetchall()
    uname_to_id = {row.uname: row.id for row in user_rows}

    # Вытащим WO, где колонка ещё пуста
    wo_rows = conn.execute(sa.text("""
        SELECT id, LOWER(TRIM(technician_name)) AS tname
        FROM work_orders
        WHERE technician_id IS NULL
    """)).fetchall()

    # Обновим по точному совпадению (посимвольно, регистр игнорим, пробелы срезаны)
    for r in wo_rows:
        tid = uname_to_id.get(r.tname)
        if tid:
            conn.execute(sa.text("""
                UPDATE work_orders
                SET technician_id = :tid
                WHERE id = :wo_id
            """), {"tid": tid, "wo_id": r.id})

    # Примечание: если у тебя есть старые WO с неточным написанием имени,
    # их можно будет добить админом вручную через UI после миграции.


def downgrade():
    # Откат: убираем FK/индекс/колонку
    with op.batch_alter_table("work_orders", schema=None) as batch_op:
        # попытка снести FK (если создавался)
        try:
            batch_op.drop_constraint("fk_work_orders_technician_id_user", type_="foreignkey")
        except Exception:
            pass
        try:
            batch_op.drop_index("ix_work_orders_technician_id")
        except Exception:
            pass
        batch_op.drop_column("technician_id")
