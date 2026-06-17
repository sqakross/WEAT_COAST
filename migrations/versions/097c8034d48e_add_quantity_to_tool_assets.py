"""add quantity to tool assets

Revision ID: 097c8034d48e
Revises: bce4b147ac5a
Create Date: 2026-06-17 12:33:33.472096

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '097c8034d48e'
down_revision = 'bce4b147ac5a'
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("tool_assets") as batch_op:
        batch_op.add_column(
            sa.Column(
                "quantity",
                sa.Integer(),
                nullable=False,
                server_default="0",
            )
        )


def downgrade():
    with op.batch_alter_table("tool_assets") as batch_op:
        batch_op.drop_column("quantity")