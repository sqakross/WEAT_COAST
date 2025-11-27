from alembic import op
import sqlalchemy as sa

revision = "20251126_add_consumed_job_ref"
down_revision = "20251124_add_tech_note_to_supplier_return_item"  # <-- поставь свою последнюю ревизию
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    insp = sa.inspect(bind)

    tables = insp.get_table_names()
    if "issued_part_record" in tables:
        cols = [c["name"] for c in insp.get_columns("issued_part_record")]
        if "consumed_job_ref" not in cols:
            op.add_column(
                "issued_part_record",
                sa.Column("consumed_job_ref", sa.String(length=64), nullable=True),
            )
            op.create_index(
                "ix_issued_part_record_consumed_job_ref",
                "issued_part_record",
                ["consumed_job_ref"],
                unique=False,
            )


def downgrade():
    bind = op.get_bind()
    insp = sa.inspect(bind)

    tables = insp.get_table_names()
    if "issued_part_record" in tables:
        cols = [c["name"] for c in insp.get_columns("issued_part_record")]
        if "consumed_job_ref" in cols:
            op.drop_index("ix_issued_part_record_consumed_job_ref", table_name="issued_part_record")
            op.drop_column("issued_part_record", "consumed_job_ref")
