"""add appliance receiving void audit fields

Revision ID: cb7e71206e30
Revises: 4893e0c10a2a
Create Date: 2026-08-29 11:20:34.375558

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'cb7e71206e30'
down_revision = '4893e0c10a2a'
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table(
        "appliance_receiving",
        schema=None,
    ) as batch_op:
        batch_op.add_column(
            sa.Column(
                "voided_at",
                sa.DateTime(),
                nullable=True,
            )
        )
        batch_op.add_column(
            sa.Column(
                "voided_by_id",
                sa.Integer(),
                nullable=True,
            )
        )
        batch_op.add_column(
            sa.Column(
                "void_reason",
                sa.Text(),
                nullable=True,
            )
        )

        batch_op.create_index(
            "ix_appliance_receiving_voided_at",
            ["voided_at"],
            unique=False,
        )
        batch_op.create_index(
            "ix_appliance_receiving_voided_by_id",
            ["voided_by_id"],
            unique=False,
        )

        batch_op.create_foreign_key(
            "fk_appliance_receiving_voided_by_id_user",
            "user",
            ["voided_by_id"],
            ["id"],
            ondelete="SET NULL",
        )


def downgrade():
    with op.batch_alter_table(
        "appliance_receiving",
        schema=None,
    ) as batch_op:
        batch_op.drop_constraint(
            "fk_appliance_receiving_voided_by_id_user",
            type_="foreignkey",
        )

        batch_op.drop_index(
            "ix_appliance_receiving_voided_by_id"
        )
        batch_op.drop_index(
            "ix_appliance_receiving_voided_at"
        )

        batch_op.drop_column("void_reason")
        batch_op.drop_column("voided_by_id")
        batch_op.drop_column("voided_at")
