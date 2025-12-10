from alembic import op
import sqlalchemy as sa

# УНИКАЛЬНЫЙ ID ДЛЯ ЭТОЙ МИГРАЦИИ
revision = "20251208_add_insurance_supplied_flags"
# ЦЕПЛЯЕМСЯ ЗА ТВОЮ ПОСЛЕДНЮЮ МИГРАЦИЮ
down_revision = "20251126_add_issued_consumption_log"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    insp = sa.inspect(bind)

    # ---------- work_order_parts: is_insurance_supplied ----------
    if "work_order_parts" in insp.get_table_names():
        cols = [c["name"] for c in insp.get_columns("work_order_parts")]
        if "is_insurance_supplied" not in cols:
            op.add_column(
                "work_order_parts",
                sa.Column(
                    "is_insurance_supplied",
                    sa.Boolean(),
                    nullable=False,
                    server_default=sa.false(),  # для старых строк
                ),
            )
            # отдельный индекс
            op.create_index(
                "ix_work_order_parts_is_insurance_supplied",
                "work_order_parts",
                ["is_insurance_supplied"],
                unique=False,
            )

    # ---------- issued_part_record: is_insurance_supplied ----------
    if "issued_part_record" in insp.get_table_names():
        cols = [c["name"] for c in insp.get_columns("issued_part_record")]
        if "is_insurance_supplied" not in cols:
            op.add_column(
                "issued_part_record",
                sa.Column(
                    "is_insurance_supplied",
                    sa.Boolean(),
                    nullable=False,
                    server_default=sa.false(),  # для существующих строк
                ),
            )
            op.create_index(
                "ix_issued_part_record_is_insurance_supplied",
                "issued_part_record",
                ["is_insurance_supplied"],
                unique=False,
            )


def downgrade():
    bind = op.get_bind()
    insp = sa.inspect(bind)

    # ---------- issued_part_record ----------
    if "issued_part_record" in insp.get_table_names():
        cols = [c["name"] for c in insp.get_columns("issued_part_record")]
        if "is_insurance_supplied" in cols:
            op.drop_index(
                "ix_issued_part_record_is_insurance_supplied",
                table_name="issued_part_record",
            )
            op.drop_column("issued_part_record", "is_insurance_supplied")

    # ---------- work_order_parts ----------
    if "work_order_parts" in insp.get_table_names():
        cols = [c["name"] for c in insp.get_columns("work_order_parts")]
        if "is_insurance_supplied" in cols:
            op.drop_index(
                "ix_work_order_parts_is_insurance_supplied",
                table_name="work_order_parts",
            )
            op.drop_column("work_order_parts", "is_insurance_supplied")
