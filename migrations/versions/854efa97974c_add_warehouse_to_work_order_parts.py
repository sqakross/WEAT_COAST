"""no-op placeholder to linearize migration chain"""

from alembic import op
import sqlalchemy as sa

revision = "854efa97974c"      # ВАЖНО: реальный id из имени файла
down_revision = "673e135df7d2"  # ← идём ПОСЛЕ твоей warehouse-миграции
branch_labels = None
depends_on = None

def upgrade():
    # ничего не делаем
    pass

def downgrade():
    # ничего не откатываем
    pass


