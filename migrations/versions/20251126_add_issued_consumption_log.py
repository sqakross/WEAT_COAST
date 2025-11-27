from alembic import op
import sqlalchemy as sa

revision = "20251126_add_issued_consumption_log"
down_revision = "20251126_add_consumed_job_ref"  # твой последний revision
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    insp = sa.inspect(bind)

    tables = insp.get_table_names()
    if "issued_consumption_log" not in tables:
        op.create_table(
            "issued_consumption_log",
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column("issued_part_id", sa.Integer, nullable=False, index=True),
            sa.Column("qty", sa.Integer, nullable=False),
            sa.Column("job_ref", sa.String(length=64), nullable=True),
            sa.Column("consumed_at", sa.DateTime, nullable=False),
            sa.Column("consumed_by", sa.String(length=120), nullable=True),
            sa.Column("note", sa.String(length=500), nullable=True),
        )

        # FK отдельно через op.create_foreign_key (SQLite нормально переварит)
        op.create_foreign_key(
            "fk_issued_consumption_log_issued_part",
            "issued_consumption_log",
            "issued_part_record",
            ["issued_part_id"],
            ["id"],
            ondelete="CASCADE",
        )

        op.create_index(
            "ix_issued_consumption_log_issued_part_id",
            "issued_consumption_log",
            ["issued_part_id"],
            unique=False,
        )
        op.create_index(
            "ix_issued_consumption_log_job_ref",
            "issued_consumption_log",
            ["job_ref"],
            unique=False,
        )
        op.create_index(
            "ix_issued_consumption_log_consumed_at",
            "issued_consumption_log",
            ["consumed_at"],
            unique=False,
        )


def downgrade():
    bind = op.get_bind()
    insp = sa.inspect(bind)

    tables = insp.get_table_names()
    if "issued_consumption_log" in tables:
        op.drop_index(
            "ix_issued_consumption_log_consumed_at",
            table_name="issued_consumption_log",
        )
        op.drop_index(
            "ix_issued_consumption_log_job_ref",
            table_name="issued_consumption_log",
        )
        op.drop_index(
            "ix_issued_consumption_log_issued_part_id",
            table_name="issued_consumption_log",
        )
        op.drop_constraint(
            "fk_issued_consumption_log_issued_part",
            "issued_consumption_log",
            type_="foreignkey",
        )
        op.drop_table("issued_consumption_log")
