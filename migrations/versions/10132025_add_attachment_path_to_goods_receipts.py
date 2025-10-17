from alembic import op
import sqlalchemy as sa

# подставь свои реальные ревизии:
revision = "10132025"
down_revision = "10122025"
branch_labels = None
depends_on = None

def upgrade():
    with op.batch_alter_table("goods_receipts") as batch:
        batch.add_column(sa.Column("attachment_path", sa.String(length=512), nullable=True))

def downgrade():
    with op.batch_alter_table("goods_receipts") as batch:
        batch.drop_column("attachment_path")
