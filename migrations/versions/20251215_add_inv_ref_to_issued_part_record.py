from alembic import op
import sqlalchemy as sa

revision = "20251215_add_inv_ref_to_issued_part_record"
down_revision = "20251214_add_po_and_invoice_fields"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    insp = sa.inspect(bind)

    if "issued_part_record" in insp.get_table_names():
        cols = [c["name"] for c in insp.get_columns("issued_part_record")]
        if "inv_ref" not in cols:
            op.add_column(
                "issued_part_record",
                sa.Column("inv_ref", sa.String(length=32), nullable=True),
            )
            op.create_index(
                "ix_issued_part_record_inv_ref",
                "issued_part_record",
                ["inv_ref"],
                unique=False,
            )


def downgrade():
    bind = op.get_bind()
    insp = sa.inspect(bind)

    if "issued_part_record" in insp.get_table_names():
        cols = [c["name"] for c in insp.get_columns("issued_part_record")]
        if "inv_ref" in cols:
            op.drop_index(
                "ix_issued_part_record_inv_ref",
                table_name="issued_part_record",
            )
            op.drop_column("issued_part_record", "inv_ref")
