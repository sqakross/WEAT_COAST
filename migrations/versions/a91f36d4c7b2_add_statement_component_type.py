"""add supplier statement component type

Revision ID: a91f36d4c7b2
Revises: 6e8ac2df9352
Create Date: 2026-09-18
"""

from alembic import op
import sqlalchemy as sa


revision = "a91f36d4c7b2"
down_revision = "6e8ac2df9352"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "supplier_statement_line_component",
        sa.Column(
            "component_type",
            sa.String(length=20),
            nullable=False,
            server_default="RETURN",
        ),
    )


def downgrade():
    with op.batch_alter_table(
        "supplier_statement_line_component"
    ) as batch_op:
        batch_op.drop_column(
            "component_type"
        )
