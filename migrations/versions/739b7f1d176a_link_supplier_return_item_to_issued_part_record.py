"""Link supplier return item to issued part record

Revision ID: 739b7f1d176a
Revises: 43d42458ba81
Create Date: 2026-07-18

"""

from alembic import op
import sqlalchemy as sa


revision = "739b7f1d176a"
down_revision = "43d42458ba81"
branch_labels = None
depends_on = None


def upgrade():

    op.add_column(
        "supplier_return_item",
        sa.Column(
            "issued_part_record_id",
            sa.Integer(),
            nullable=True,
        ),
    )

    op.create_index(
        "ix_supplier_return_item_issued_part_record_id",
        "supplier_return_item",
        ["issued_part_record_id"],
    )

    # SQLite не любит ALTER TABLE ADD CONSTRAINT.
    # Поэтому FK пока НЕ создаем.
    # При переходе на PostgreSQL/MySQL можно будет добавить отдельной migration.


def downgrade():

    op.drop_index(
        "ix_supplier_return_item_issued_part_record_id",
        table_name="supplier_return_item",
    )

    op.drop_column(
        "supplier_return_item",
        "issued_part_record_id",
    )