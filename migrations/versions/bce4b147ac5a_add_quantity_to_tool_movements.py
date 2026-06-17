from alembic import op
import sqlalchemy as sa

revision = 'bce4b147ac5a'
down_revision = '50ce3abeb7be'
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("tool_movements") as batch_op:
        batch_op.add_column(
            sa.Column(
                "quantity",
                sa.Integer(),
                nullable=False,
                server_default="1",
            )
        )


def downgrade():
    with op.batch_alter_table("tool_movements") as batch_op:
        batch_op.drop_column("quantity")