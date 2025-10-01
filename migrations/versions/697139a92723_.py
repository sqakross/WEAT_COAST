"""legacy migration turned into no-op, chained after 854efa97974c"""

from alembic import op
import sqlalchemy as sa

revision = '697139a92723'
down_revision = '854efa97974c'  # ← теперь ЗА пустышкой 854…
branch_labels = None
depends_on = None

def upgrade():
    pass

def downgrade():
    pass

