"""Add stock and consumption tracking fields"""

from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = "11072025_add_stock"
down_revision = "102720251_applied_qty"
branch_labels = None
depends_on = None


def upgrade():
    # === issued_part_records ===
    op.add_column("issued_part_records", sa.Column("consumed_qty", sa.Integer(), nullable=True))
    op.add_column("issued_part_records", sa.Column("consumed_flag", sa.Boolean(), nullable=False, server_default=sa.false()))
    op.add_column("issued_part_records", sa.Column("consumed_at", sa.DateTime(), nullable=True))
    op.add_column("issued_part_records", sa.Column("consumed_by", sa.String(length=120), nullable=True))
    op.add_column("issued_part_records", sa.Column("consumed_note", sa.String(length=500), nullable=True))

    # === issued_batches ===
    op.add_column("issued_batches", sa.Column("is_stock", sa.Boolean(), nullable=False, server_default=sa.false()))
    op.add_column("issued_batches", sa.Column("consumed_flag", sa.Boolean(), nullable=False, server_default=sa.false()))
    op.add_column("issued_batches", sa.Column("consumed_at", sa.DateTime(), nullable=True))
    op.add_column("issued_batches", sa.Column("consumed_by", sa.String(length=120), nullable=True))
    op.add_column("issued_batches", sa.Column("consumed_note", sa.String(length=500), nullable=True))

    # Убираем server_default (чтобы поля не вставлялись с дефолтом при каждом апдейте)
    op.execute("ALTER TABLE issued_part_records ALTER COLUMN consumed_flag DROP DEFAULT")
    op.execute("ALTER TABLE issued_batches ALTER COLUMN is_stock DROP DEFAULT")
    op.execute("ALTER TABLE issued_batches ALTER COLUMN consumed_flag DROP DEFAULT")


def downgrade():
    # issued_part_records
    op.drop_column("issued_part_records", "consumed_note")
    op.drop_column("issued_part_records", "consumed_by")
    op.drop_column("issued_part_records", "consumed_at")
    op.drop_column("issued_part_records", "consumed_flag")
    op.drop_column("issued_part_records", "consumed_qty")

    # issued_batches
    op.drop_column("issued_batches", "consumed_note")
    op.drop_column("issued_batches", "consumed_by")
    op.drop_column("issued_batches", "consumed_at")
    op.drop_column("issued_batches", "consumed_flag")
    op.drop_column("issued_batches", "is_stock")
