"""add re-receive existing appliance link

Revision ID: b7c4e2a91f30
Revises: a91f36d4c7b2
Create Date: 2026-09-30 09:18:18
"""

from alembic import op
import sqlalchemy as sa


revision = "b7c4e2a91f30"
down_revision = "a91f36d4c7b2"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table(
        "appliance_receiving_line"
    ) as batch_op:

        batch_op.add_column(
            sa.Column(
                "existing_appliance_unit_id",
                sa.Integer(),
                nullable=True,
            )
        )

        batch_op.create_foreign_key(
            "fk_appliance_receiving_line_existing_unit",
            "appliance_unit",
            ["existing_appliance_unit_id"],
            ["id"],
            ondelete="RESTRICT",
        )

        batch_op.create_index(
            "ix_appliance_receiving_line_existing_appliance_unit_id",
            ["existing_appliance_unit_id"],
            unique=False,
        )


def downgrade():
    with op.batch_alter_table(
        "appliance_receiving_line"
    ) as batch_op:

        batch_op.drop_index(
            "ix_appliance_receiving_line_existing_appliance_unit_id"
        )

        batch_op.drop_constraint(
            "fk_appliance_receiving_line_existing_unit",
            type_="foreignkey",
        )

        batch_op.drop_column(
            "existing_appliance_unit_id"
        )
