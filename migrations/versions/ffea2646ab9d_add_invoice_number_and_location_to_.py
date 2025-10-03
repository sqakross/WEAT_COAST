from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = "ffea2646ab9d"
down_revision = "854efa97974c"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("issued_part_record", schema=None) as batch_op:
        batch_op.add_column(sa.Column("invoice_number", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("location", sa.String(length=120), nullable=True))
        batch_op.create_index("ix_issued_part_record_invoice_number", ["invoice_number"], unique=False)


def downgrade():
    with op.batch_alter_table("issued_part_record", schema=None) as batch_op:
        batch_op.drop_index("ix_issued_part_record_invoice_number")
        batch_op.drop_column("location")
        batch_op.drop_column("invoice_number")

