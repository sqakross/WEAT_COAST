"""Add supplier return reconciliation fields

Revision ID: 43d42458ba81
Revises: fa00164a1011
Create Date: 2026-07-18 11:32:34.248339

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "43d42458ba81"
down_revision = "fa00164a1011"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "supplier_return_batch",
        sa.Column(
            "supplier_credit_document",
            sa.String(length=64),
            nullable=True,
        ),
    )

    op.add_column(
        "supplier_return_batch",
        sa.Column(
            "supplier_credit_date",
            sa.Date(),
            nullable=True,
        ),
    )

    op.add_column(
        "supplier_return_batch",
        sa.Column(
            "reconciled_at",
            sa.DateTime(),
            nullable=True,
        ),
    )

    op.create_index(
        "ix_supplier_return_batch_supplier_credit_document",
        "supplier_return_batch",
        ["supplier_credit_document"],
        unique=False,
    )


def downgrade():
    op.drop_index(
        "ix_supplier_return_batch_supplier_credit_document",
        table_name="supplier_return_batch",
    )

    op.drop_column(
        "supplier_return_batch",
        "reconciled_at",
    )

    op.drop_column(
        "supplier_return_batch",
        "supplier_credit_date",
    )

    op.drop_column(
        "supplier_return_batch",
        "supplier_credit_document",
    )